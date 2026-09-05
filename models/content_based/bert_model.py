"""
models/content_based/bert_model.py
------------------------------------
BERT-based semantic content recommender.

BERT (Bidirectional Encoder Representations from Transformers) produces
dense semantic embeddings that capture MEANING, not just keyword overlap.

Why BERT over TF-IDF?
- TF-IDF: "Science Fiction" and "Sci-Fi" are completely different terms
- BERT:   "Science Fiction" and "Sci-Fi" have nearly identical embeddings

This matters for libraries because:
- Book titles use varied language for similar concepts
- Genre descriptions overlap semantically
- BERT captures this nuance that TF-IDF misses entirely

We use sentence-transformers/all-MiniLM-L6-v2:
- Fast: 14,200 sentences/second on CPU
- Small: 80MB model size
- Strong: Outperforms larger models on semantic similarity tasks
- Free: No API key needed, runs locally
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Sentence Transformers import with graceful fallback ───────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
    logger.info("sentence-transformers available — BERT model enabled")
except ImportError:
    BERT_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers"
    )


class BERTRecommender:
    """
    Semantic content-based recommender using BERT sentence embeddings.

    Each book gets a 384-dimensional embedding vector that encodes
    its semantic meaning. Patron profiles are built by averaging
    embeddings of borrowed books. Recommendations are the books
    with highest cosine similarity to the patron profile.

    Advantage over TF-IDF: captures semantic similarity, not just
    keyword overlap. A patron who reads "AI and Machine Learning"
    will get recommendations for "Deep Learning" and "Neural Networks"
    even if those exact words don't appear in their history.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32):
        """
        Args:
            model_name: Sentence transformer model to use
            batch_size: Encoding batch size (larger = faster, more memory)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.encoder = None
        self.book_embeddings = None
        self.book_metadata = None
        self.book_index = None
        self.book_index_reverse = None
        self.embedding_dim = None
        self.is_fitted = False

    def fit(self, lms_df: pd.DataFrame) -> "BERTRecommender":
        """
        Encodes all books in the catalog using BERT embeddings.

        This is the slow step — encoding 1000 books takes ~30 seconds
        on CPU. In production, embeddings are computed once and cached.

        Args:
            lms_df: Raw LMS borrowing records

        Returns:
            self (fitted model)
        """
        if not BERT_AVAILABLE:
            raise ImportError("sentence-transformers not installed")

        logger.info(f"Loading BERT encoder: {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name)

        # Build book metadata
        self.book_metadata = lms_df.groupby("book_id").agg(
            book_title=("book_title", "first"),
            genre=("genre", "first"),
            total_borrows=("patron_id", "count")
        ).reset_index()

        # Build index mappings
        self.book_index = {
            bid: i for i, bid in enumerate(self.book_metadata["book_id"])
        }
        self.book_index_reverse = {v: k for k, v in self.book_index.items()}

        # Build content text for encoding
        # BERT understands natural language so we write a proper description
        content_texts = [
            f"A {row['genre']} book titled {row['book_title']}"
            for _, row in self.book_metadata.iterrows()
        ]

        logger.info(f"Encoding {len(content_texts)} books with BERT...")
        self.book_embeddings = self.encoder.encode(
            content_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.embedding_dim = self.book_embeddings.shape[1]
        self.is_fitted = True

        logger.info(
            f"BERT encoding complete: {len(self.book_metadata)} books, "
            f"{self.embedding_dim}-dimensional embeddings"
        )
        return self

    def build_patron_profile(self, patron_id: str,
                              lms_df: pd.DataFrame,
                              recency_weight: bool = True) -> np.ndarray:
        """
        Builds a patron semantic profile from their borrowing history.

        With recency_weight=True, more recently borrowed books get
        higher weight in the profile. This reflects that current
        taste is more predictive than what was read years ago.

        Args:
            patron_id:      Patron to build profile for
            lms_df:         LMS borrowing records
            recency_weight: Weight recent borrows more heavily

        Returns:
            numpy array of shape (embedding_dim,) — patron profile
        """
        patron_borrows = lms_df[lms_df["patron_id"] == patron_id].copy()
        patron_books = patron_borrows["book_id"].unique()
        patron_books_in_catalog = [b for b in patron_books if b in self.book_index]

        if not patron_books_in_catalog:
            return None

        indices = [self.book_index[b] for b in patron_books_in_catalog]
        borrowed_embeddings = self.book_embeddings[indices]

        if recency_weight and "checkout_date" in patron_borrows.columns:
            # More recent borrows get higher weight
            patron_borrows = patron_borrows.sort_values("checkout_date")
            n = len(patron_books_in_catalog)
            weights = np.linspace(0.5, 1.0, n)
            weights = weights / weights.sum()
            profile = np.average(borrowed_embeddings, axis=0, weights=weights)
        else:
            profile = borrowed_embeddings.mean(axis=0)

        return profile

    def recommend(self, patron_id: str,
                  lms_df: pd.DataFrame,
                  n_recommendations: int = 10,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Recommends books using semantic similarity to patron profile.

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

        # Cosine similarity between patron profile and all book embeddings
        similarities = cosine_similarity(
            profile.reshape(1, -1),
            self.book_embeddings
        ).flatten()

        # Exclude already-borrowed books
        if exclude_seen:
            seen_books = lms_df[lms_df["patron_id"] == patron_id]["book_id"].unique()
            for book in seen_books:
                if book in self.book_index:
                    similarities[self.book_index[book]] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

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

    def get_similar_books(self, book_id: str,
                           n_similar: int = 10) -> pd.DataFrame:
        """
        Finds semantically similar books using BERT embeddings.

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
        book_embedding = self.book_embeddings[book_idx].reshape(1, -1)

        similarities = cosine_similarity(
            book_embedding, self.book_embeddings
        ).flatten()
        similarities[book_idx] = -1

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

    def save(self, path: str) -> None:
        """Saves embeddings and metadata — not the encoder (download separately)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "book_embeddings": self.book_embeddings,
            "book_metadata": self.book_metadata,
            "book_index": self.book_index,
            "book_index_reverse": self.book_index_reverse,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"BERT model data saved to {path}")

    def load(self, path: str) -> "BERTRecommender":
        """Loads embeddings and metadata, reloads encoder."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.book_embeddings = data["book_embeddings"]
        self.book_metadata = data["book_metadata"]
        self.book_index = data["book_index"]
        self.book_index_reverse = data["book_index_reverse"]
        self.model_name = data["model_name"]
        self.embedding_dim = data["embedding_dim"]
        self.encoder = SentenceTransformer(self.model_name)
        self.is_fitted = True
        logger.info(f"BERT model loaded from {path}")
        return self


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python models/content_based/bert_model.py
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "ingestion"
    ))
    from lms_connector import load_lms_data

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=2000)

    if not BERT_AVAILABLE:
        print("\nsentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
    else:
        print("Fitting BERT model (this takes ~30 seconds on CPU)...")
        model = BERTRecommender(model_name="all-MiniLM-L6-v2", batch_size=32)
        model.fit(lms_df)

        test_patron = lms_df["patron_id"].iloc[0]
        print(f"\n── Recommendations for patron {test_patron} ──")
        recs = model.recommend(test_patron, lms_df, n_recommendations=5)
        print(recs)

        test_book = lms_df["book_id"].iloc[0]
        print(f"\n── Semantically similar books to {test_book} ──")
        similar = model.get_similar_books(test_book, n_similar=5)
        print(similar)

        print(f"\nEmbedding dimension: {model.embedding_dim}")
        print(f"Books encoded: {len(model.book_metadata)}")