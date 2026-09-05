"""
models/collaborative/svd_model.py
----------------------------------
Singular Value Decomposition (SVD) collaborative filtering model.

SVD is a classical matrix factorization approach that decomposes
the patron-book interaction matrix into three matrices:
    R ≈ U × Σ × V^T

Where:
- U: patron latent factors (patron embeddings)
- Σ: singular values (importance of each factor)
- V: book latent factors (book embeddings)

Why SVD alongside ALS?
- SVD is deterministic — same result every run, easier to debug
- ALS handles missing values better (implicit feedback)
- SVD is faster for smaller datasets
- In the hybrid engine, their predictions are ensembled together
  giving better coverage than either model alone

Implemented using sklearn's TruncatedSVD which is memory-efficient
for sparse matrices — no need to densify the full matrix.
"""

import pandas as pd
import numpy as np
import logging
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVDModel:
    """
    SVD-based collaborative filtering using TruncatedSVD.

    TruncatedSVD keeps only the top-k singular values/vectors,
    which captures the most important latent structure while
    ignoring noise in the interaction data.
    """

    def __init__(self, n_components: int = 50,
                 n_iter: int = 10,
                 seed: int = 42):
        """
        Args:
            n_components: Number of latent dimensions to keep
            n_iter:       Number of SVD iterations (for convergence)
            seed:         Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.seed = seed
        self.svd = TruncatedSVD(
            n_components=n_components,
            n_iter=n_iter,
            random_state=seed
        )
        self.patron_factors = None
        self.book_factors = None
        self.patron_index = None
        self.book_index = None
        self.patron_index_reverse = None
        self.book_index_reverse = None
        self.explained_variance_ratio = None
        self.is_fitted = False

    def fit(self, interaction_matrix: csr_matrix,
            patron_index: dict,
            book_index: dict) -> "SVDModel":
        """
        Fits SVD on the patron-book interaction matrix.

        The key difference from ALS:
        SVD fits on the full matrix at once (not iteratively per user/item).
        This makes it faster but less flexible for implicit feedback.

        Args:
            interaction_matrix: Sparse patron-book interaction matrix
            patron_index:       Dict mapping patron_id to matrix row
            book_index:         Dict mapping book_id to matrix column

        Returns:
            self (fitted model)
        """
        self.patron_index = patron_index
        self.book_index = book_index
        self.patron_index_reverse = {v: k for k, v in patron_index.items()}
        self.book_index_reverse = {v: k for k, v in book_index.items()}

        n_users, n_items = interaction_matrix.shape
        logger.info(
            f"Training SVD: {n_users} patrons, {n_items} books, "
            f"{self.n_components} components"
        )

        # Fit SVD — patron_factors are the transformed patron embeddings
        self.patron_factors = self.svd.fit_transform(interaction_matrix)

        # Book factors are the right singular vectors (V^T transposed)
        self.book_factors = self.svd.components_.T

        # How much variance each component explains
        self.explained_variance_ratio = self.svd.explained_variance_ratio_
        total_explained = self.explained_variance_ratio.sum()

        logger.info(
            f"SVD training complete — "
            f"{self.n_components} components explain "
            f"{total_explained:.1%} of variance"
        )

        self.is_fitted = True
        return self

    def recommend(self, patron_id: str,
                  n_recommendations: int = 10,
                  exclude_seen: bool = True,
                  interaction_matrix: csr_matrix = None) -> pd.DataFrame:
        """
        Generates top-N book recommendations for a patron.

        Recommendation score = dot product of patron embedding
        and book embedding. Higher score = better match.

        Args:
            patron_id:          Patron to recommend for
            n_recommendations:  Number of books to return
            exclude_seen:       Exclude already-borrowed books
            interaction_matrix: Needed to identify seen books

        Returns:
            pd.DataFrame with book_id and predicted_score columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend()")

        if patron_id not in self.patron_index:
            logger.warning(f"Patron {patron_id} not in training data")
            return pd.DataFrame(columns=["book_id", "predicted_score"])

        patron_idx = self.patron_index[patron_id]
        patron_vector = self.patron_factors[patron_idx]

        # Score all books via dot product with patron embedding
        scores = self.book_factors @ patron_vector

        # Exclude already-borrowed books
        if exclude_seen and interaction_matrix is not None:
            seen_items = interaction_matrix[patron_idx].nonzero()[1]
            scores[seen_items] = -np.inf

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n_recommendations]

        recommendations = pd.DataFrame({
            "book_id": [self.book_index_reverse[i] for i in top_indices],
            "predicted_score": scores[top_indices].round(4)
        })

        return recommendations

    def get_similar_books(self, book_id: str,
                          n_similar: int = 10) -> pd.DataFrame:
        """
        Finds books most similar to a given book in latent space.

        Useful for "readers also enjoyed" style recommendations.
        Similarity measured by cosine similarity of book embeddings.

        Args:
            book_id:   The book to find similar books for
            n_similar: Number of similar books to return

        Returns:
            pd.DataFrame with book_id and similarity_score columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling get_similar_books()")

        if book_id not in self.book_index:
            logger.warning(f"Book {book_id} not in training data")
            return pd.DataFrame(columns=["book_id", "similarity_score"])

        book_idx = self.book_index[book_id]
        book_vector = self.book_factors[book_idx].reshape(1, -1)

        # Cosine similarity between this book and all others
        normalized_factors = normalize(self.book_factors)
        normalized_book = normalize(book_vector)
        similarities = (normalized_factors @ normalized_book.T).flatten()

        # Exclude the book itself
        similarities[book_idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n_similar]

        return pd.DataFrame({
            "book_id": [self.book_index_reverse[i] for i in top_indices],
            "similarity_score": similarities[top_indices].round(4)
        })

    def save(self, path: str) -> None:
        """Saves the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SVD model saved to {path}")

    @staticmethod
    def load(path: str) -> "SVDModel":
        """Loads a fitted model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"SVD model loaded from {path}")
        return model


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python models/collaborative/svd_model.py
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "ingestion"
    ))
    from lms_connector import load_lms_data
    from als_model import build_interaction_matrix

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=3000)

    print("Building interaction matrix...")
    matrix, patron_idx, book_idx = build_interaction_matrix(lms_df)

    print("Training SVD model...")
    model = SVDModel(n_components=20, n_iter=10)
    model.fit(matrix, patron_idx, book_idx)

    # Test recommendations
    test_patron = list(patron_idx.keys())[0]
    print(f"\n── Recommendations for patron {test_patron} ──")
    recs = model.recommend(
        test_patron, n_recommendations=5, interaction_matrix=matrix
    )
    print(recs)

    # Test similar books
    test_book = list(book_idx.keys())[0]
    print(f"\n── Books similar to {test_book} ──")
    similar = model.get_similar_books(test_book, n_similar=5)
    print(similar)

    print(f"\nVariance explained: {model.explained_variance_ratio.sum():.1%}")
    print(f"Patron factors: {model.patron_factors.shape}")
    print(f"Book factors:   {model.book_factors.shape}")