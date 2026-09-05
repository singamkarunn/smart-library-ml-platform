"""
features/content_features.py
-----------------------------
Extracts content-based features from book metadata.

Content features describe the BOOKS themselves — not patron behavior,
but what the books are about. These feed directly into the content-based
recommendation model which recommends books similar to what a patron
has borrowed before, even for patrons with sparse borrowing history
(the cold-start problem).

Two approaches implemented:
1. TF-IDF: Fast, interpretable, works well for short text
2. BERT embeddings: Slower, captures semantic meaning, stronger signal

Output: A book-level DataFrame where each row is one book
        and columns are content feature vectors.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Book Metadata Builder ─────────────────────────────────────────────────
def build_book_metadata(lms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a clean book metadata table from LMS borrowing records.

    In production this would come from a dedicated book catalog API
    (ISBN database, library catalog system). Here we derive it from
    the borrowing records since that's what the synthetic data has.

    Args:
        lms_df: Raw LMS borrowing records

    Returns:
        pd.DataFrame with one row per unique book and metadata columns
    """
    logger.info("Building book metadata from LMS records...")

    # Deduplicate to get one row per book
    book_metadata = lms_df.groupby("book_id").agg(
        book_title=("book_title", "first"),
        genre=("genre", "first"),
        total_borrows=("patron_id", "count"),
        unique_borrowers=("patron_id", "nunique"),
        avg_loan_duration=("loan_duration_days", "mean"),
        avg_renewals=("times_renewed", "mean"),
        overdue_rate=("loan_status", lambda x: (x == "overdue").mean())
    ).reset_index()

    book_metadata["avg_loan_duration"] = book_metadata["avg_loan_duration"].round(2)
    book_metadata["avg_renewals"] = book_metadata["avg_renewals"].round(3)
    book_metadata["overdue_rate"] = book_metadata["overdue_rate"].round(3)

    logger.info(f"Built metadata for {len(book_metadata)} unique books")
    return book_metadata


# ── TF-IDF Content Features ───────────────────────────────────────────────
def extract_tfidf_features(book_metadata: pd.DataFrame,
                            max_features: int = 100) -> pd.DataFrame:
    """
    Extracts TF-IDF features from book titles and genres.

    TF-IDF works by:
    1. Tokenizing the text (splitting into words)
    2. Counting how often each word appears in each book description
    3. Downweighting words that appear in every book (common words)
    4. Upweighting words that are specific to certain books

    Why TF-IDF for books?
    - Fast to compute — scales to 250K+ catalog items
    - Interpretable — you can see which words drive similarity
    - Works well when book descriptions are short

    Args:
        book_metadata: Book metadata DataFrame from build_book_metadata
        max_features:  Max number of TF-IDF vocabulary features

    Returns:
        pd.DataFrame with book_id and TF-IDF feature columns
    """
    logger.info(f"Extracting TF-IDF features (max_features={max_features})...")

    # Combine title and genre into a single text field
    # Genre is repeated 3x to give it more weight than individual title words
    book_metadata["content_text"] = (
        book_metadata["book_title"] + " " +
        book_metadata["genre"] + " " +
        book_metadata["genre"] + " " +
        book_metadata["genre"]
    )

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(book_metadata["content_text"])

    # Convert sparse matrix to DataFrame
    feature_names = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names
    )

    tfidf_df.insert(0, "book_id", book_metadata["book_id"].values)

    logger.info(f"TF-IDF features extracted: {tfidf_df.shape[1] - 1} features for {len(tfidf_df)} books")
    return tfidf_df, vectorizer


# ── Genre Encoding Features ───────────────────────────────────────────────
def extract_genre_encoding(book_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one-hot encoded genre features and popularity signals.

    One-hot encoding converts categorical genre labels into binary
    columns that ML models can process directly. Combined with
    popularity metrics, these give the model book-level signals
    beyond just text similarity.

    Args:
        book_metadata: Book metadata DataFrame

    Returns:
        pd.DataFrame with book_id, one-hot genre columns, and popularity features
    """
    logger.info("Extracting genre encoding and popularity features...")

    # One-hot encode genre
    genre_dummies = pd.get_dummies(
        book_metadata["genre"],
        prefix="genre"
    ).astype(int)

    features = pd.concat([
        book_metadata[["book_id"]],
        genre_dummies
    ], axis=1)

    # Popularity tier — top 10%, middle 80%, bottom 10% by borrow count
    borrow_quantiles = book_metadata["total_borrows"].quantile([0.1, 0.9])
    features["popularity_tier"] = pd.cut(
        book_metadata["total_borrows"],
        bins=[-1, borrow_quantiles[0.1], borrow_quantiles[0.9], float("inf")],
        labels=["low", "medium", "high"]
    ).astype(str)

    # Normalized popularity score (0 to 1)
    max_borrows = book_metadata["total_borrows"].max()
    features["popularity_score"] = (
        book_metadata["total_borrows"] / max_borrows
    ).round(4).values

    # Engagement score — books with high renewals are "sticky"
    features["engagement_score"] = book_metadata["avg_renewals"].round(3).values

    # Overdue rate — proxy for how compelling the book is
    # (people renew or go overdue for books they want to keep reading)
    features["overdue_rate"] = book_metadata["overdue_rate"].values

    logger.info(f"Genre encoding extracted: {len(features.columns) - 1} features")
    return features


# ── Master Content Feature Builder ────────────────────────────────────────
def build_content_features(lms_df: pd.DataFrame,
                            tfidf_max_features: int = 50) -> dict:
    """
    Builds all content-based features and returns them as a dict.

    Returns a dict rather than a single DataFrame because TF-IDF
    produces high-dimensional sparse features that are used differently
    from the compact genre/popularity features.

    Args:
        lms_df:              LMS borrowing records
        tfidf_max_features:  Max TF-IDF vocabulary size

    Returns:
        dict with keys:
            "metadata"  — raw book metadata
            "tfidf"     — TF-IDF feature matrix
            "encoding"  — genre encoding + popularity features
            "vectorizer"— fitted TF-IDF vectorizer (for inference)
    """
    logger.info("Building full content feature set...")

    metadata_df = build_book_metadata(lms_df)
    tfidf_df, vectorizer = extract_tfidf_features(metadata_df, tfidf_max_features)
    encoding_df = extract_genre_encoding(metadata_df)

    logger.info(
        f"Content features complete — "
        f"{len(metadata_df)} books, "
        f"{tfidf_df.shape[1] - 1} TF-IDF features, "
        f"{encoding_df.shape[1] - 1} encoding features"
    )

    return {
        "metadata": metadata_df,
        "tfidf": tfidf_df,
        "encoding": encoding_df,
        "vectorizer": vectorizer
    }


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python features/content_features.py
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingestion"))

    from lms_connector import load_lms_data

    print("Loading LMS data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=5000)

    print("Building content features...")
    content = build_content_features(lms_df, tfidf_max_features=50)

    print("\n── Book Metadata Sample ──")
    print(content["metadata"].head(3))

    print("\n── TF-IDF Features ──")
    print(f"Shape: {content['tfidf'].shape}")
    print(content["tfidf"].iloc[:3, :6])

    print("\n── Genre Encoding Sample ──")
    print(content["encoding"].head(3))

    print("\n── Vectorizer Vocabulary Size ──")
    print(f"{len(content['vectorizer'].vocabulary_)} terms")