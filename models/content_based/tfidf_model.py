"""
models/content_based/tfidf_model.py
-------------------------------------
TF-IDF content-based recommendation model.

Content-based filtering recommends books similar to what a patron
has already borrowed — based on the CONTENT of the books (title,
genre) rather than what other patrons borrowed.

Why content-based alongside collaborative filtering?
- Solves the cold-start problem for NEW books (no borrow history yet)
- Explains recommendations clearly ("because you borrowed Science Fiction")
- Works for patrons with very niche tastes that CF can't model well
- In the hybrid engine, CF + content-based together outperform either alone

TF-IDF approach:
1. Build a text representation for each book (title + genre)
2. Compute TF-IDF vectors for all books
3. For a patron, aggregate TF-IDF vectors of their borrowed books
4. Find books with highest cosine similarity to the patron's profile
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFRecommender:
    """
    Content-based recommender using TF-IDF book representations.

    The core idea: build a "taste profile" for each patron by
    averaging the TF-IDF vectors of all books they've borrowed.
    Then recommend books whose TF-IDF vectors are closest to
    that patron profile in cosine similarity space.
    """

    def __init__(self, max_features: int = 500,
                 ngram_range: tuple = (1, 2),
                 min_df: int = 1):
        """
        Args:
            max_features:  Max TF-IDF vocabulary size
            ngram_range:   (1,2) includes unigrams and bigrams
            min_df:        Minimum document frequency for a term
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=min_df
        )
        self.book_vectors = None
        self.book_metadata = None
        self.book_index = None
        self.book_index_reverse = None
        self.is_fitted = False

    def fit(self, lms_df: pd.DataFrame) -> "TFIDFRecommender":
        """
        Fits TF-IDF vectorizer on book catalog and builds book vectors.

        Args:
            lms_df: Raw LMS borrowing records

        Returns:
            self (fitted model)
        """
        logger.info("Fitting TF-IDF content-based model...")

        # Build book metadata
        self.book_metadata = lms_df.groupby("book_id").agg(
            book_title=("book_title", "first"),
            genre=("genre", "first"),
            total_borrows=("patron_id", "count")
        ).reset_index()

        # Build content text — genre repeated for emphasis
        self.book_metadata["content_text"] = (
            self.book_metadata["book_title"] + " " +
            self.book_metadata["genre"] + " " +
            self.book_metadata["genre"] + " " +
            self.book_metadata["genre"]
        )

        # Build index mappings
        self.book_index = {
            bid: i for i, bid in enumerate(self.book_metadata["book_id"])
        }
        self.book_index_reverse = {v: k for k, v in self.book_index.items()}

        # Fit and transform all books
        self.book_vectors = self.vectorizer.fit_transform(
            self.book_metadata["content_text"]
        )

        self.is_fitted = True
        logger.info(
            f"TF-IDF fitted: {len(self.book_metadata)} books, "
            f"{self.book_vectors.shape[1]} features"
        )
        return self

    def build_patron_profile(self, patron_id: str,
                              lms_df: pd.DataFrame) -> np.ndarray:
        """
        Builds a patron taste profile by averaging borrowed book vectors.

        The profile vector represents what the patron tends to read —
        genres and title keywords weighted by how much they appear
        across the patron's borrowing history.

        Args:
            patron_id: The patron to build a profile for
            lms_df:    LMS borrowing records

        Returns:
            numpy array representing the patron's taste profile
        """
        patron_books = lms_df[lms_df["patron_id"] == patron_id]["book_id"].unique()
        patron_books_in_catalog = [b for b in patron_books if b in self.book_index]

        if not patron_books_in_catalog:
            return None

        # Get TF-IDF vectors for all borrowed books
        indices = [self.book_index[b] for b in patron_books_in_catalog]
        borrowed_vectors = self.book_vectors[indices]

        # Patron profile = mean of all borrowed book vectors
        # This captures the "average taste" across their reading history
        profile = np.asarray(borrowed_vectors.mean(axis=0))
        return profile

    def recommend(self, patron_id: str,
                  lms_df: pd.DataFrame,
                  n_recommendations: int = 10,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Recommends books based on patron's content taste profile.

        Args:
            patron_id:          Patron to recommend for
            lms_df:             LMS borrowing records
            n_recommendations:  Number of books to return
            exclude_seen:       Exclude already-borrowed books

        Returns:
            pd.DataFrame with book_id, similarity_score, genre columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend()")

        profile = self.build_patron_profile(patron_id, lms_df)

        if profile is None:
            logger.warning(f"No borrowing history for patron {patron_id}")
            return pd.DataFrame(columns=["book_id", "similarity_score", "genre"])

        # Compute cosine similarity between profile and all books
        similarities = cosine_similarity(profile, self.book_vectors).flatten()

        # Exclude already-borrowed books
        if exclude_seen:
            seen_books = lms_df[lms_df["patron_id"] == patron_id]["book_id"].unique()
            for book in seen_books:
                if book in self.book_index:
                    similarities[self.book_index[book]] = -1

        # Get top-N
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        recommendations = pd.DataFrame({
            "book_id": [self.book_index_reverse[i] for i in top_indices],
            "similarity_score": similarities[top_indices].round(4),
            "genre": [
                self.book_metadata.loc[
                    self.book_metadata["book_id"] == self.book_index_reverse[i],
                    "genre"
                ].values[0]
                for i in top_indices
            ]
        })

        return recommendations

    def get_similar_books(self, book_id: str,
                           n_similar: int = 10) -> pd.DataFrame:
        """
        Finds books most similar to a given book by content.

        Args:
            book_id:   Source book
            n_similar: Number of similar books to return

        Returns:
            pd.DataFrame with book_id, similarity_score, genre columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling get_similar_books()")

        if book_id not in self.book_index:
            logger.warning(f"Book {book_id} not in catalog")
            return pd.DataFrame(columns=["book_id", "similarity_score", "genre"])

        book_idx = self.book_index[book_id]
        book_vector = self.book_vectors[book_idx]

        similarities = cosine_similarity(book_vector, self.book_vectors).flatten()
        similarities[book_idx] = -1  # exclude itself

        top_indices = np.argsort(similarities)[::-1][:n_similar]

        return pd.DataFrame({
            "book_id": [self.book_index_reverse[i] for i in top_indices],
            "similarity_score": similarities[top_indices].round(4),
            "genre": [
                self.book_metadata.loc[
                    self.book_metadata["book_id"] == self.book_index_reverse[i],
                    "genre"
                ].values[0]
                for i in top_indices
            ]
        })

    def explain_recommendation(self, patron_id: str,
                                book_id: str,
                                lms_df: pd.DataFrame,
                                n_terms: int = 5) -> str:
        """
        Explains WHY a book was recommended to a patron.

        Identifies the top TF-IDF terms driving the similarity
        between the patron's profile and the recommended book.

        Args:
            patron_id: The patron
            book_id:   The recommended book
            lms_df:    LMS borrowing records
            n_terms:   Number of explanation terms to show

        Returns:
            Human-readable explanation string
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explaining")

        profile = self.build_patron_profile(patron_id, lms_df)
        if profile is None:
            return "No borrowing history available for explanation"

        if book_id not in self.book_index:
            return f"Book {book_id} not in catalog"

        book_idx = self.book_index[book_id]
        book_vector = np.asarray(self.book_vectors[book_idx].todense()).flatten()

        # Terms driving similarity = element-wise product of profile and book vector
        feature_names = self.vectorizer.get_feature_names_out()
        overlap = profile.flatten() * book_vector
        top_term_indices = np.argsort(overlap)[::-1][:n_terms]
        top_terms = [feature_names[i] for i in top_term_indices if overlap[i] > 0]

        book_genre = self.book_metadata.loc[
            self.book_metadata["book_id"] == book_id, "genre"
        ].values[0]

        if top_terms:
            return (f"Recommended because your reading history matches "
                    f"'{book_genre}' content — key signals: {', '.join(top_terms)}")
        else:
            return f"Recommended based on genre match: {book_genre}"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"TF-IDF model saved to {path}")

    @staticmethod
    def load(path: str) -> "TFIDFRecommender":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"TF-IDF model loaded from {path}")
        return model


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python models/content_based/tfidf_model.py
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "ingestion"
    ))
    from lms_connector import load_lms_data

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=3000)

    print("Fitting TF-IDF model...")
    model = TFIDFRecommender(max_features=200, ngram_range=(1, 2))
    model.fit(lms_df)

    test_patron = lms_df["patron_id"].iloc[0]
    print(f"\n── Recommendations for patron {test_patron} ──")
    recs = model.recommend(test_patron, lms_df, n_recommendations=5)
    print(recs)

    test_book = lms_df["book_id"].iloc[0]
    print(f"\n── Similar books to {test_book} ──")
    similar = model.get_similar_books(test_book, n_similar=5)
    print(similar)

    print(f"\n── Explanation ──")
    if len(recs) > 0:
        explanation = model.explain_recommendation(
            test_patron, recs["book_id"].iloc[0], lms_df
        )
        print(explanation)