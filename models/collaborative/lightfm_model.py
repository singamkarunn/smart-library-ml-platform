"""
models/collaborative/lightfm_model.py
--------------------------------------
LightFM hybrid collaborative filtering model.

LightFM is unique among collaborative filtering models because it
can use BOTH interaction data AND side features (patron demographics,
book genres) simultaneously. This makes it particularly powerful for:

1. Cold-start: New patrons with no history can still get recommendations
   based on their age group or membership type
2. New books: Books with no borrows can be recommended based on genre
3. Hybrid signal: Combines collaborative + content signals in one model

Why LightFM for libraries?
- Libraries have rich metadata (genres, patron types) that pure CF ignores
- New patron cold-start is a real problem in library systems
- LightFM's WARP loss is designed for implicit feedback (borrows, not ratings)

Note: LightFM requires separate installation on Python 3.11+
      If not installed, this module falls back to SVD recommendations.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LightFM import with graceful fallback ─────────────────────────────────
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import precision_at_k, recall_at_k
    LIGHTFM_AVAILABLE = True
    logger.info("LightFM available — full model enabled")
except ImportError:
    LIGHTFM_AVAILABLE = False
    logger.warning(
        "LightFM not installed — using SVD fallback. "
        "Install with: pip install lightfm==1.16"
    )


# ── Feature Builder ───────────────────────────────────────────────────────
def build_lightfm_dataset(lms_df: pd.DataFrame) -> tuple:
    """
    Builds the LightFM Dataset object with patron and book features.

    LightFM Dataset handles the mapping between raw IDs and internal
    indices, and constructs the sparse interaction and feature matrices.

    Patron features used:
    - Age group (18-25, 26-35, 36-50, 51-65, 65+)
    - Membership type (basic, standard, premium)

    Book features used:
    - Genre (Fiction, Non-Fiction, etc.)

    Args:
        lms_df: Raw LMS borrowing records

    Returns:
        tuple of (dataset, interactions, weights, patron_features, book_features)
    """
    if not LIGHTFM_AVAILABLE:
        raise ImportError("LightFM is not installed")

    logger.info("Building LightFM dataset with side features...")

    dataset = Dataset()

    # Define all patron and book features upfront
    patron_feature_names = [
        f"age_group:{ag}" for ag in lms_df["patron_age_group"].unique()
    ] + [
        f"membership:{mt}" for mt in lms_df["patron_membership_type"].unique()
    ]

    book_feature_names = [
        f"genre:{g}" for g in lms_df["genre"].unique()
    ]

    # Fit the dataset — registers all IDs and feature names
    dataset.fit(
        users=lms_df["patron_id"].unique(),
        items=lms_df["book_id"].unique(),
        user_features=patron_feature_names,
        item_features=book_feature_names
    )

    # Build interaction matrix from borrowing records
    (interactions, weights) = dataset.build_interactions([
        (row["patron_id"], row["book_id"], 1 + 0.5 * row["times_renewed"])
        for _, row in lms_df.iterrows()
    ])

    # Build patron feature matrix
    patron_feature_data = lms_df.drop_duplicates("patron_id")[
        ["patron_id", "patron_age_group", "patron_membership_type"]
    ]
    patron_features = dataset.build_user_features([
        (
            row["patron_id"],
            [f"age_group:{row['patron_age_group']}",
             f"membership:{row['patron_membership_type']}"]
        )
        for _, row in patron_feature_data.iterrows()
    ])

    # Build book feature matrix
    book_feature_data = lms_df.drop_duplicates("book_id")[["book_id", "genre"]]
    book_features = dataset.build_item_features([
        (row["book_id"], [f"genre:{row['genre']}"])
        for _, row in book_feature_data.iterrows()
    ])

    logger.info(
        f"Dataset built: {interactions.shape[0]} patrons, "
        f"{interactions.shape[1]} books, "
        f"{interactions.nnz} interactions"
    )

    return dataset, interactions, weights, patron_features, book_features


# ── LightFM Model ─────────────────────────────────────────────────────────
class LightFMModel:
    """
    LightFM wrapper with training, evaluation, and recommendation.

    Uses WARP (Weighted Approximate-Rank Pairwise) loss which is
    specifically designed for implicit feedback recommendation —
    it optimizes for ranking (is the right book in the top-10?)
    rather than rating prediction (is the score exactly right?).
    """

    def __init__(self, n_components: int = 32,
                 loss: str = "warp",
                 n_epochs: int = 30,
                 learning_rate: float = 0.05,
                 seed: int = 42):
        """
        Args:
            n_components:  Number of latent embedding dimensions
            loss:          Loss function — "warp" for implicit, "bpr" alternative
            n_epochs:      Training epochs
            learning_rate: SGD learning rate
            seed:          Random seed
        """
        self.n_components = n_components
        self.loss = loss
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.dataset = None
        self.interactions = None
        self.patron_features = None
        self.book_features = None
        self.patron_id_map = None
        self.book_id_map = None
        self.is_fitted = False

    def fit(self, lms_df: pd.DataFrame) -> "LightFMModel":
        """
        Trains LightFM on borrowing records with side features.

        Args:
            lms_df: Raw LMS borrowing records

        Returns:
            self (fitted model)
        """
        if not LIGHTFM_AVAILABLE:
            raise ImportError("LightFM is not installed")

        (self.dataset,
         self.interactions,
         weights,
         self.patron_features,
         self.book_features) = build_lightfm_dataset(lms_df)

        # Get ID mappings for recommendation lookup
        mappings = self.dataset.mapping()
        self.patron_id_map = mappings[0]   # patron_id -> internal index
        self.book_id_map = mappings[2]     # book_id -> internal index
        self.book_id_map_reverse = {v: k for k, v in self.book_id_map.items()}

        logger.info(
            f"Training LightFM: loss={self.loss}, "
            f"n_components={self.n_components}, "
            f"n_epochs={self.n_epochs}"
        )

        self.model = LightFM(
            no_components=self.n_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            random_state=self.seed
        )

        self.model.fit(
            self.interactions,
            user_features=self.patron_features,
            item_features=self.book_features,
            epochs=self.n_epochs,
            num_threads=2,
            verbose=False
        )

        self.is_fitted = True
        logger.info("LightFM training complete")
        return self

    def evaluate(self) -> dict:
        """
        Evaluates the model using Precision@K and Recall@K.

        Returns:
            dict with precision_at_10 and recall_at_10 scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluating")

        p_at_10 = precision_at_k(
            self.model, self.interactions,
            user_features=self.patron_features,
            item_features=self.book_features,
            k=10
        ).mean()

        r_at_10 = recall_at_k(
            self.model, self.interactions,
            user_features=self.patron_features,
            item_features=self.book_features,
            k=10
        ).mean()

        metrics = {
            "precision_at_10": round(float(p_at_10), 4),
            "recall_at_10": round(float(r_at_10), 4)
        }

        logger.info(f"LightFM evaluation — P@10: {metrics['precision_at_10']}, R@10: {metrics['recall_at_10']}")
        return metrics

    def recommend(self, patron_id: str,
                  n_recommendations: int = 10,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Generates top-N recommendations for a patron.

        Args:
            patron_id:          Patron to recommend for
            n_recommendations:  Number of books to return
            exclude_seen:       Exclude already-borrowed books

        Returns:
            pd.DataFrame with book_id and predicted_score columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend()")

        if patron_id not in self.patron_id_map:
            logger.warning(f"Patron {patron_id} not in training data")
            return pd.DataFrame(columns=["book_id", "predicted_score"])

        patron_idx = self.patron_id_map[patron_id]
        n_books = len(self.book_id_map)

        scores = self.model.predict(
            patron_idx,
            np.arange(n_books),
            user_features=self.patron_features,
            item_features=self.book_features
        )

        if exclude_seen:
            seen = self.interactions[patron_idx].nonzero()[1]
            scores[seen] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_recommendations]

        return pd.DataFrame({
            "book_id": [self.book_id_map_reverse[i] for i in top_indices],
            "predicted_score": scores[top_indices].round(4)
        })

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)