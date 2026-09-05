"""
models/collaborative/als_model.py
----------------------------------
Alternating Least Squares (ALS) collaborative filtering model.

ALS is a matrix factorization technique that decomposes the
patron-book interaction matrix into two lower-dimensional matrices:
- User factors (patron embeddings)
- Item factors (book embeddings)

Why ALS for libraries?
- Handles implicit feedback well (borrows, not ratings)
- Scales to large catalogs (250K+ books)
- Fast to train with alternating optimization
- Cold-start handled by falling back to popularity

The core idea: if patron A and patron B both borrowed books X and Y,
they probably share taste — so recommend books that B borrowed to A.
"""

import pandas as pd
import numpy as np
import logging
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Interaction Matrix Builder ────────────────────────────────────────────
def build_interaction_matrix(lms_df: pd.DataFrame) -> tuple:
    """
    Builds a sparse patron-book interaction matrix from borrowing records.

    Why sparse?
    A library with 500 patrons and 1000 books has a 500x1000 matrix
    with 500,000 cells. But most patrons only borrow a tiny fraction
    of all books — so 99%+ of cells are zero. Sparse matrices store
    only the non-zero values, saving memory and computation.

    Interaction values:
    - 1 base for each borrow
    - +0.5 for each renewal (stronger signal of engagement)
    - Capped at 5 to prevent heavy borrowers from dominating

    Args:
        lms_df: Raw LMS borrowing records

    Returns:
        tuple of (sparse_matrix, patron_index, book_index)
        where patron_index and book_index map IDs to matrix positions
    """
    logger.info("Building patron-book interaction matrix...")

    # Compute interaction strength per patron-book pair
    interactions = lms_df.groupby(["patron_id", "book_id"]).agg(
        borrow_count=("loan_status", "count"),
        total_renewals=("times_renewed", "sum")
    ).reset_index()

    # Weighted interaction score
    interactions["interaction_score"] = (
        interactions["borrow_count"] +
        0.5 * interactions["total_renewals"]
    ).clip(upper=5)

    # Build index mappings
    patron_index = {pid: i for i, pid in enumerate(interactions["patron_id"].unique())}
    book_index = {bid: i for i, bid in enumerate(interactions["book_id"].unique())}

    # Build sparse matrix
    rows = interactions["patron_id"].map(patron_index)
    cols = interactions["book_id"].map(book_index)
    data = interactions["interaction_score"].values

    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(patron_index), len(book_index))
    )

    logger.info(
        f"Interaction matrix built: {sparse_matrix.shape} "
        f"({sparse_matrix.nnz} non-zero entries, "
        f"{100 * sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.2f}% density)"
    )

    return sparse_matrix, patron_index, book_index


# ── ALS Model ─────────────────────────────────────────────────────────────
class ALSModel:
    """
    Alternating Least Squares collaborative filtering model.

    Implements ALS from scratch using scipy for sparse operations.
    In production you would use implicit.als.AlternatingLeastSquares
    which is GPU-accelerated. This implementation demonstrates the
    core algorithm clearly.
    """

    def __init__(self, n_factors: int = 50,
                 n_iterations: int = 20,
                 regularization: float = 0.01,
                 seed: int = 42):
        """
        Args:
            n_factors:      Number of latent factors (embedding dimensions)
            n_iterations:   Number of ALS iterations
            regularization: L2 regularization to prevent overfitting
            seed:           Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.seed = seed
        self.user_factors = None
        self.item_factors = None
        self.patron_index = None
        self.book_index = None
        self.patron_index_reverse = None
        self.book_index_reverse = None
        self.is_fitted = False

    def fit(self, interaction_matrix: csr_matrix,
            patron_index: dict,
            book_index: dict) -> "ALSModel":
        """
        Trains the ALS model on the interaction matrix.

        ALS alternates between:
        1. Fixing item factors, solving for optimal user factors
        2. Fixing user factors, solving for optimal item factors

        This continues for n_iterations until convergence.

        Args:
            interaction_matrix: Sparse patron-book interaction matrix
            patron_index:       Dict mapping patron_id to matrix row
            book_index:         Dict mapping book_id to matrix column

        Returns:
            self (fitted model)
        """
        np.random.seed(self.seed)

        n_users, n_items = interaction_matrix.shape
        self.patron_index = patron_index
        self.book_index = book_index
        self.patron_index_reverse = {v: k for k, v in patron_index.items()}
        self.book_index_reverse = {v: k for k, v in book_index.items()}

        # Initialize factors randomly
        self.user_factors = np.random.normal(
            scale=1.0 / self.n_factors,
            size=(n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=1.0 / self.n_factors,
            size=(n_items, self.n_factors)
        )

        logger.info(
            f"Training ALS: {n_users} patrons, {n_items} books, "
            f"{self.n_factors} factors, {self.n_iterations} iterations"
        )

        R = interaction_matrix.toarray()
        lambda_I = self.regularization * np.eye(self.n_factors)

        for iteration in range(self.n_iterations):
            # Fix item factors, solve for user factors
            for u in range(n_users):
                rated_items = R[u] > 0
                if rated_items.sum() == 0:
                    continue
                X = self.item_factors[rated_items]
                ratings = R[u, rated_items]
                A = X.T @ X + lambda_I
                b = X.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)

            # Fix user factors, solve for item factors
            for i in range(n_items):
                rated_users = R[:, i] > 0
                if rated_users.sum() == 0:
                    continue
                X = self.user_factors[rated_users]
                ratings = R[rated_users, i]
                A = X.T @ X + lambda_I
                b = X.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                # Compute training loss for monitoring
                predicted = self.user_factors @ self.item_factors.T
                loss = np.mean((R - predicted) ** 2)
                logger.info(f"  Iteration {iteration + 1}/{self.n_iterations} — MSE: {loss:.4f}")

        self.is_fitted = True
        logger.info("ALS training complete")
        return self

    def recommend(self, patron_id: str,
                  n_recommendations: int = 10,
                  exclude_seen: bool = True,
                  interaction_matrix: csr_matrix = None) -> pd.DataFrame:
        """
        Generates top-N book recommendations for a patron.

        Args:
            patron_id:          The patron to recommend for
            n_recommendations:  Number of books to recommend
            exclude_seen:       Whether to exclude already-borrowed books
            interaction_matrix: Original matrix (needed to exclude seen)

        Returns:
            pd.DataFrame with book_id and predicted_score columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend()")

        if patron_id not in self.patron_index:
            logger.warning(f"Patron {patron_id} not in training data — no recommendations")
            return pd.DataFrame(columns=["book_id", "predicted_score"])

        patron_idx = self.patron_index[patron_id]
        patron_vector = self.user_factors[patron_idx]

        # Score all books
        scores = self.item_factors @ patron_vector

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

    def save(self, path: str) -> None:
        """Saves the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"ALS model saved to {path}")

    @staticmethod
    def load(path: str) -> "ALSModel":
        """Loads a fitted model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"ALS model loaded from {path}")
        return model


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python models/collaborative/als_model.py
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ingestion"))

    from lms_connector import load_lms_data

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=3000)

    print("Building interaction matrix...")
    matrix, patron_idx, book_idx = build_interaction_matrix(lms_df)

    print("Training ALS model...")
    model = ALSModel(n_factors=20, n_iterations=10, regularization=0.01)
    model.fit(matrix, patron_idx, book_idx)

    # Test recommendation for first patron
    test_patron = list(patron_idx.keys())[0]
    print(f"\n── Recommendations for patron {test_patron} ──")
    recs = model.recommend(test_patron, n_recommendations=5,
                           interaction_matrix=matrix)
    print(recs)
    print(f"\nModel factors: user={model.user_factors.shape}, item={model.item_factors.shape}")