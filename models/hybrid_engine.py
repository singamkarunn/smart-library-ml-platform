"""
models/hybrid_engine.py
------------------------
Hybrid recommendation engine combining all models into one system.

The hybrid engine is the core of the Smart Library ML Platform.
It combines signals from four models:

1. ALS (collaborative filtering)     — what similar patrons borrowed
2. SVD (collaborative filtering)     — latent taste structure
3. TF-IDF (content-based)            — keyword/genre similarity
4. BERT (content-based)              — semantic similarity

Why hybrid over any single model?
- ALS/SVD fail for new patrons (cold-start) — content-based fills the gap
- Content-based fails for niche titles — collaborative fills the gap
- Ensemble consistently outperforms any individual model
- Different models capture different signals — combining them is additive

Fusion strategy: weighted score averaging
Each model produces a ranked list with scores. We normalize each model's
scores to [0,1], apply configurable weights, and average them.
The final ranking is by combined score.

Rules layer: business logic applied on top of model scores
- Boost new books (recently added to catalog)
- Boost books in patron's top genre
- Cap recommendations from any single genre (diversity constraint)
"""

import pandas as pd
import numpy as np
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Path setup for imports ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "ingestion"))
sys.path.insert(0, os.path.join(BASE_DIR, "collaborative"))
sys.path.insert(0, os.path.join(BASE_DIR, "content_based"))


class HybridRecommendationEngine:
    """
    Hybrid recommendation engine combining collaborative and content-based models.

    Architecture:
        ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
        │   ALS   │  │   SVD   │  │  TF-IDF  │  │   BERT   │
        └────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘
             │            │            │              │
             └────────────┴────────────┴──────────────┘
                                  │
                         Score Normalization
                                  │
                         Weighted Averaging
                                  │
                           Rules Layer
                                  │
                        Final Recommendations
    """

    def __init__(self,
                 als_weight: float = 0.35,
                 svd_weight: float = 0.25,
                 tfidf_weight: float = 0.20,
                 bert_weight: float = 0.20,
                 diversity_cap: int = 3,
                 new_book_boost: float = 0.05):
        """
        Args:
            als_weight:      Weight for ALS model scores (0-1)
            svd_weight:      Weight for SVD model scores (0-1)
            tfidf_weight:    Weight for TF-IDF model scores (0-1)
            bert_weight:     Weight for BERT model scores (0-1)
            diversity_cap:   Max recommendations from any single genre
            new_book_boost:  Score boost for recently added books
        """
        assert abs(als_weight + svd_weight + tfidf_weight + bert_weight - 1.0) < 1e-6, \
            "Model weights must sum to 1.0"

        self.als_weight = als_weight
        self.svd_weight = svd_weight
        self.tfidf_weight = tfidf_weight
        self.bert_weight = bert_weight
        self.diversity_cap = diversity_cap
        self.new_book_boost = new_book_boost

        # Model instances
        self.als_model = None
        self.svd_model = None
        self.tfidf_model = None
        self.bert_model = None

        # Training data reference
        self.lms_df = None
        self.interaction_matrix = None
        self.patron_index = None
        self.book_index = None
        self.book_metadata = None

        self.is_fitted = False

    def fit(self, lms_df: pd.DataFrame,
            use_bert: bool = True) -> "HybridRecommendationEngine":
        """
        Trains all component models on LMS borrowing data.

        Args:
            lms_df:    Raw LMS borrowing records
            use_bert:  Whether to train BERT (slow but more accurate)

        Returns:
            self (fitted engine)
        """
        from als_model import ALSModel, build_interaction_matrix
        from svd_model import SVDModel
        from tfidf_model import TFIDFRecommender

        logger.info("Fitting Hybrid Recommendation Engine...")
        self.lms_df = lms_df

        # Build book metadata reference
        self.book_metadata = lms_df.groupby("book_id").agg(
            genre=("genre", "first"),
            total_borrows=("patron_id", "count")
        ).reset_index()

        # Build interaction matrix (shared by ALS and SVD)
        logger.info("Building interaction matrix...")
        self.interaction_matrix, self.patron_index, self.book_index = \
            build_interaction_matrix(lms_df)

        # Train ALS
        logger.info("Training ALS model...")
        self.als_model = ALSModel(n_factors=20, n_iterations=10)
        self.als_model.fit(
            self.interaction_matrix, self.patron_index, self.book_index
        )

        # Train SVD
        logger.info("Training SVD model...")
        self.svd_model = SVDModel(n_components=20)
        self.svd_model.fit(
            self.interaction_matrix, self.patron_index, self.book_index
        )

        # Train TF-IDF
        logger.info("Training TF-IDF model...")
        self.tfidf_model = TFIDFRecommender(max_features=200)
        self.tfidf_model.fit(lms_df)

        # Train BERT (optional — slow on CPU)
        if use_bert:
            try:
                from bert_model import BERTRecommender, BERT_AVAILABLE
                if BERT_AVAILABLE:
                    logger.info("Training BERT model...")
                    self.bert_model = BERTRecommender()
                    self.bert_model.fit(lms_df)
                else:
                    logger.warning("BERT not available — using TF-IDF weight only")
                    self._redistribute_bert_weight()
            except Exception as e:
                logger.warning(f"BERT training failed: {e} — redistributing weight")
                self._redistribute_bert_weight()
        else:
            logger.info("BERT skipped — redistributing weight to other models")
            self._redistribute_bert_weight()

        self.is_fitted = True
        logger.info("Hybrid engine training complete")
        return self

    def _redistribute_bert_weight(self) -> None:
        """Redistributes BERT weight to other models when BERT unavailable."""
        extra = self.bert_weight
        self.bert_weight = 0.0
        total = self.als_weight + self.svd_weight + self.tfidf_weight
        self.als_weight += extra * (self.als_weight / total)
        self.svd_weight += extra * (self.svd_weight / total)
        self.tfidf_weight += extra * (self.tfidf_weight / total)
        logger.info(
            f"Weights redistributed — ALS: {self.als_weight:.2f}, "
            f"SVD: {self.svd_weight:.2f}, TF-IDF: {self.tfidf_weight:.2f}"
        )

    def _normalize_scores(self, recommendations: pd.DataFrame,
                           score_col: str = "predicted_score") -> pd.DataFrame:
        """
        Normalizes model scores to [0, 1] range.

        Why normalize?
        ALS scores might be in range [0, 5] while cosine similarities
        are in [0, 1]. Without normalization, ALS would dominate simply
        because its scores are numerically larger, not because it's better.
        """
        if len(recommendations) == 0:
            return recommendations

        min_score = recommendations[score_col].min()
        max_score = recommendations[score_col].max()

        if max_score == min_score:
            recommendations["normalized_score"] = 1.0
        else:
            recommendations["normalized_score"] = (
                (recommendations[score_col] - min_score) /
                (max_score - min_score)
            )
        return recommendations

    def _get_model_recommendations(self, patron_id: str,
                                    n: int = 50) -> dict:
        """
        Gets top-N recommendations from each model.

        We request 50 candidates per model (more than final N)
        so the fusion layer has enough candidates to work with
        after applying diversity constraints.

        Returns:
            dict mapping model_name -> pd.DataFrame of recommendations
        """
        model_recs = {}

        # ALS recommendations
        if self.als_model and patron_id in self.patron_index:
            als_recs = self.als_model.recommend(
                patron_id, n_recommendations=n,
                interaction_matrix=self.interaction_matrix
            )
            if len(als_recs) > 0:
                als_recs = self._normalize_scores(als_recs)
                als_recs["model"] = "als"
                model_recs["als"] = als_recs

        # SVD recommendations
        if self.svd_model and patron_id in self.svd_model.patron_index:
            svd_recs = self.svd_model.recommend(
                patron_id, n_recommendations=n,
                interaction_matrix=self.interaction_matrix
            )
            if len(svd_recs) > 0:
                svd_recs = self._normalize_scores(svd_recs)
                svd_recs["model"] = "svd"
                model_recs["svd"] = svd_recs

        # TF-IDF recommendations
        if self.tfidf_model:
            tfidf_recs = self.tfidf_model.recommend(
                patron_id, self.lms_df, n_recommendations=n
            )
            if len(tfidf_recs) > 0:
                tfidf_recs = tfidf_recs.rename(
                    columns={"similarity_score": "predicted_score"}
                )
                tfidf_recs = self._normalize_scores(tfidf_recs)
                tfidf_recs["model"] = "tfidf"
                model_recs["tfidf"] = tfidf_recs

        # BERT recommendations
        if self.bert_model and self.bert_weight > 0:
            bert_recs = self.bert_model.recommend(
                patron_id, self.lms_df, n_recommendations=n
            )
            if len(bert_recs) > 0:
                bert_recs = bert_recs.rename(
                    columns={"similarity_score": "predicted_score"}
                )
                bert_recs = self._normalize_scores(bert_recs)
                bert_recs["model"] = "bert"
                model_recs["bert"] = bert_recs

        return model_recs

    def _fuse_scores(self, model_recs: dict) -> pd.DataFrame:
        """
        Fuses model scores using weighted average fusion.

        For each book that appears in any model's recommendations:
        1. Collect normalized scores from each model
        2. Apply model weight
        3. Sum weighted scores (missing models contribute 0)
        4. Rank by combined score

        Args:
            model_recs: Dict of model_name -> recommendations DataFrame

        Returns:
            pd.DataFrame with book_id and combined_score, ranked
        """
        weight_map = {
            "als": self.als_weight,
            "svd": self.svd_weight,
            "tfidf": self.tfidf_weight,
            "bert": self.bert_weight
        }

        # Collect all candidate books
        all_books = set()
        for recs in model_recs.values():
            all_books.update(recs["book_id"].tolist())

        if not all_books:
            return pd.DataFrame(columns=["book_id", "combined_score"])

        # Build score lookup per model
        score_lookup = {}
        for model_name, recs in model_recs.items():
            score_lookup[model_name] = dict(
                zip(recs["book_id"], recs["normalized_score"])
            )

        # Compute weighted combined score
        combined = []
        for book_id in all_books:
            score = sum(
                weight_map.get(model_name, 0) *
                score_lookup[model_name].get(book_id, 0)
                for model_name in model_recs.keys()
            )
            combined.append({"book_id": book_id, "combined_score": score})

        result = pd.DataFrame(combined)
        result = result.sort_values("combined_score", ascending=False)
        result["combined_score"] = result["combined_score"].round(4)
        return result.reset_index(drop=True)

    def _apply_rules(self, fused_df: pd.DataFrame,
                     patron_id: str,
                     n_recommendations: int) -> pd.DataFrame:
        """
        Applies business rules on top of fused model scores.

        Rules applied:
        1. Genre diversity cap — no more than diversity_cap books per genre
        2. Popularity boost — slightly boost high-borrow books for cold patrons
        3. Genre boost — boost books in patron's historically preferred genre

        Args:
            fused_df:           Score-fused candidate books
            patron_id:          Patron being recommended for
            n_recommendations:  Final number to return

        Returns:
            pd.DataFrame with final recommendations
        """
        # Join genre information
        result = fused_df.merge(
            self.book_metadata[["book_id", "genre", "total_borrows"]],
            on="book_id", how="left"
        )

        # Popularity boost (small — keeps model signal dominant)
        max_borrows = result["total_borrows"].max()
        if max_borrows > 0:
            result["combined_score"] += (
                self.new_book_boost * result["total_borrows"] / max_borrows
            )

        # Genre boost — find patron's top genre
        patron_borrows = self.lms_df[self.lms_df["patron_id"] == patron_id]
        if len(patron_borrows) > 0:
            top_genre = patron_borrows["genre"].mode()[0]
            result.loc[result["genre"] == top_genre, "combined_score"] += 0.02

        # Re-sort after boosts
        result = result.sort_values("combined_score", ascending=False)

        # Genre diversity cap
        final_recs = []
        genre_counts = {}

        for _, row in result.iterrows():
            genre = row.get("genre", "Unknown")
            if genre_counts.get(genre, 0) < self.diversity_cap:
                final_recs.append(row)
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if len(final_recs) >= n_recommendations:
                break

        return pd.DataFrame(final_recs)[
            ["book_id", "combined_score", "genre"]
        ].reset_index(drop=True)

    def recommend(self, patron_id: str,
                  n_recommendations: int = 10) -> pd.DataFrame:
        """
        Main recommendation method — runs full hybrid pipeline.

        Pipeline:
        1. Get candidates from each model
        2. Normalize scores per model
        3. Fuse with weighted average
        4. Apply business rules (diversity, boosts)
        5. Return top-N

        Args:
            patron_id:          Patron to recommend for
            n_recommendations:  Number of books to return

        Returns:
            pd.DataFrame with book_id, combined_score, genre columns
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before calling recommend()")

        logger.info(f"Generating hybrid recommendations for {patron_id}...")

        # Step 1: Get candidates from all models
        model_recs = self._get_model_recommendations(patron_id, n=50)

        if not model_recs:
            logger.warning(f"No recommendations generated for {patron_id}")
            return pd.DataFrame(columns=["book_id", "combined_score", "genre"])

        # Step 2 & 3: Normalize and fuse
        fused = self._fuse_scores(model_recs)

        # Step 4 & 5: Apply rules and return top-N
        final = self._apply_rules(fused, patron_id, n_recommendations)

        logger.info(
            f"Generated {len(final)} recommendations for {patron_id} "
            f"using {list(model_recs.keys())} models"
        )
        return final

    def explain(self, patron_id: str, book_id: str) -> str:
        """
        Explains why a book was recommended.

        Args:
            patron_id: The patron
            book_id:   The recommended book

        Returns:
            Human-readable explanation string
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before explaining")

        explanations = []

        # Check if collaborative models would recommend this book
        if self.als_model and patron_id in self.patron_index:
            als_recs = self.als_model.recommend(
                patron_id, n_recommendations=20,
                interaction_matrix=self.interaction_matrix
            )
            if book_id in als_recs["book_id"].values:
                explanations.append("patrons with similar borrowing history enjoyed this book")

        # Get content explanation from TF-IDF
        if self.tfidf_model:
            content_exp = self.tfidf_model.explain_recommendation(
                patron_id, book_id, self.lms_df
            )
            explanations.append(content_exp)

        if explanations:
            return "Recommended because: " + "; and ".join(explanations)
        else:
            return f"Recommended based on your reading profile"

    def get_model_weights(self) -> dict:
        """Returns current model weights."""
        return {
            "als": self.als_weight,
            "svd": self.svd_weight,
            "tfidf": self.tfidf_weight,
            "bert": self.bert_weight
        }


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python models/hybrid_engine.py
if __name__ == "__main__":
    from lms_connector import load_lms_data

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=3000)

    print("\nInitializing Hybrid Engine...")
    engine = HybridRecommendationEngine(
        als_weight=0.35,
        svd_weight=0.25,
        tfidf_weight=0.20,
        bert_weight=0.20
    )

    print("\nFitting all models (this takes ~1 minute)...")
    engine.fit(lms_df, use_bert=True)

    print(f"\nModel weights: {engine.get_model_weights()}")

    test_patron = lms_df["patron_id"].iloc[0]
    print(f"\n── Hybrid Recommendations for {test_patron} ──")
    recs = engine.recommend(test_patron, n_recommendations=10)
    print(recs)

    print(f"\n── Explanation ──")
    if len(recs) > 0:
        explanation = engine.explain(test_patron, recs["book_id"].iloc[0])
        print(explanation)

    print(f"\n── Genre Distribution in Recommendations ──")
    print(recs["genre"].value_counts())