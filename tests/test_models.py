"""
tests/test_models.py
---------------------
Unit tests for recommendation models.

Tests verify that:
- Models train without errors
- Recommendations have correct shape and types
- Edge cases (unknown patron, empty catalog) are handled
- Score normalization and ranking work correctly

Run: python -m pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "collaborative"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "content_based"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from lms_connector import load_lms_data


# ── Shared fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lms_df():
    """Small LMS dataset shared across all model tests."""
    return load_lms_data(source="synthetic", n_transactions=1000, seed=42)


@pytest.fixture(scope="module")
def interaction_matrix_data(lms_df):
    """Shared interaction matrix for collaborative model tests."""
    from als_model import build_interaction_matrix
    matrix, patron_idx, book_idx = build_interaction_matrix(lms_df)
    return matrix, patron_idx, book_idx


@pytest.fixture(scope="module")
def trained_als(lms_df, interaction_matrix_data):
    """Pre-trained ALS model shared across tests."""
    from als_model import ALSModel
    matrix, patron_idx, book_idx = interaction_matrix_data
    model = ALSModel(n_factors=10, n_iterations=5, seed=42)
    model.fit(matrix, patron_idx, book_idx)
    return model, matrix, patron_idx, book_idx


@pytest.fixture(scope="module")
def trained_svd(lms_df, interaction_matrix_data):
    """Pre-trained SVD model shared across tests."""
    from svd_model import SVDModel
    matrix, patron_idx, book_idx = interaction_matrix_data
    model = SVDModel(n_components=10, n_iter=5, seed=42)
    model.fit(matrix, patron_idx, book_idx)
    return model, matrix, patron_idx, book_idx


@pytest.fixture(scope="module")
def trained_tfidf(lms_df):
    """Pre-trained TF-IDF model shared across tests."""
    from tfidf_model import TFIDFRecommender
    model = TFIDFRecommender(max_features=50)
    model.fit(lms_df)
    return model


# ── Interaction Matrix Tests ──────────────────────────────────────────────

class TestInteractionMatrix:

    def test_matrix_shape_correct(self, lms_df, interaction_matrix_data):
        """Matrix shape matches number of unique patrons and books."""
        matrix, patron_idx, book_idx = interaction_matrix_data
        assert matrix.shape[0] == len(patron_idx)
        assert matrix.shape[1] == len(book_idx)

    def test_matrix_values_positive(self, interaction_matrix_data):
        """All interaction scores are positive."""
        matrix, _, _ = interaction_matrix_data
        assert (matrix.data > 0).all()

    def test_matrix_values_capped(self, interaction_matrix_data):
        """Interaction scores are capped at 5."""
        matrix, _, _ = interaction_matrix_data
        assert (matrix.data <= 5).all()

    def test_patron_index_maps_all_patrons(self, lms_df, interaction_matrix_data):
        """Patron index contains all unique patrons from LMS data."""
        _, patron_idx, _ = interaction_matrix_data
        lms_patrons = set(lms_df["patron_id"].unique())
        assert lms_patrons == set(patron_idx.keys())

    def test_book_index_maps_all_books(self, lms_df, interaction_matrix_data):
        """Book index contains all unique books from LMS data."""
        _, _, book_idx = interaction_matrix_data
        lms_books = set(lms_df["book_id"].unique())
        assert lms_books == set(book_idx.keys())


# ── ALS Model Tests ───────────────────────────────────────────────────────

class TestALSModel:

    def test_model_is_fitted_after_training(self, trained_als):
        """Model reports is_fitted=True after training."""
        model, _, _, _ = trained_als
        assert model.is_fitted is True

    def test_user_factors_correct_shape(self, trained_als, interaction_matrix_data):
        """User factors have shape (n_patrons, n_factors)."""
        model, _, patron_idx, _ = trained_als
        assert model.user_factors.shape == (len(patron_idx), model.n_factors)

    def test_item_factors_correct_shape(self, trained_als, interaction_matrix_data):
        """Item factors have shape (n_books, n_factors)."""
        model, _, _, book_idx = trained_als
        assert model.item_factors.shape == (len(book_idx), model.n_factors)

    def test_recommend_returns_dataframe(self, trained_als, lms_df):
        """recommend() returns a pandas DataFrame."""
        model, matrix, patron_idx, _ = trained_als
        patron_id = list(patron_idx.keys())[0]
        recs = model.recommend(patron_id, n_recommendations=5,
                               interaction_matrix=matrix)
        assert isinstance(recs, pd.DataFrame)

    def test_recommend_correct_columns(self, trained_als, lms_df):
        """Recommendations have book_id and predicted_score columns."""
        model, matrix, patron_idx, _ = trained_als
        patron_id = list(patron_idx.keys())[0]
        recs = model.recommend(patron_id, n_recommendations=5,
                               interaction_matrix=matrix)
        assert "book_id" in recs.columns
        assert "predicted_score" in recs.columns

    def test_recommend_respects_n_recommendations(self, trained_als):
        """Returns at most n_recommendations results."""
        model, matrix, patron_idx, _ = trained_als
        patron_id = list(patron_idx.keys())[0]
        for n in [1, 5, 10]:
            recs = model.recommend(patron_id, n_recommendations=n,
                                   interaction_matrix=matrix)
            assert len(recs) <= n

    def test_recommend_excludes_seen_books(self, trained_als, lms_df):
        """When exclude_seen=True, already-borrowed books not recommended."""
        model, matrix, patron_idx, book_idx = trained_als
        patron_id = list(patron_idx.keys())[0]

        # Get books the patron has already borrowed
        seen_books = set(
            lms_df[lms_df["patron_id"] == patron_id]["book_id"].unique()
        )

        recs = model.recommend(patron_id, n_recommendations=20,
                               exclude_seen=True,
                               interaction_matrix=matrix)
        rec_books = set(recs["book_id"].tolist())

        # No overlap between seen and recommended
        assert len(seen_books & rec_books) == 0

    def test_recommend_unknown_patron_returns_empty(self, trained_als):
        """Unknown patron returns empty DataFrame, not an error."""
        model, matrix, _, _ = trained_als
        recs = model.recommend("UNKNOWN_PATRON_XYZ", n_recommendations=5,
                               interaction_matrix=matrix)
        assert len(recs) == 0

    def test_recommend_before_fitting_raises(self):
        """Calling recommend before fit raises RuntimeError."""
        from als_model import ALSModel
        model = ALSModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.recommend("P00001", n_recommendations=5)

    def test_scores_are_ranked_descending(self, trained_als):
        """Recommendation scores are in descending order."""
        model, matrix, patron_idx, _ = trained_als
        patron_id = list(patron_idx.keys())[0]
        recs = model.recommend(patron_id, n_recommendations=10,
                               interaction_matrix=matrix)
        scores = recs["predicted_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ── SVD Model Tests ───────────────────────────────────────────────────────

class TestSVDModel:

    def test_model_is_fitted_after_training(self, trained_svd):
        model, _, _, _ = trained_svd
        assert model.is_fitted is True

    def test_explained_variance_between_0_and_1(self, trained_svd):
        """Explained variance ratio is between 0 and 1."""
        model, _, _, _ = trained_svd
        total = model.explained_variance_ratio.sum()
        assert 0 < total <= 1.0

    def test_patron_factors_correct_shape(self, trained_svd, interaction_matrix_data):
        model, _, patron_idx, _ = trained_svd
        assert model.patron_factors.shape[0] == len(patron_idx)
        assert model.patron_factors.shape[1] == model.n_components

    def test_recommend_returns_correct_structure(self, trained_svd):
        model, matrix, patron_idx, _ = trained_svd
        patron_id = list(patron_idx.keys())[0]
        recs = model.recommend(patron_id, n_recommendations=5,
                               interaction_matrix=matrix)
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 5
        assert "book_id" in recs.columns

    def test_get_similar_books_returns_results(self, trained_svd):
        """get_similar_books returns expected structure."""
        model, _, _, book_idx = trained_svd
        book_id = list(book_idx.keys())[0]
        similar = model.get_similar_books(book_id, n_similar=5)
        assert isinstance(similar, pd.DataFrame)
        assert "book_id" in similar.columns
        assert "similarity_score" in similar.columns
        assert len(similar) <= 5

    def test_similar_books_excludes_self(self, trained_svd):
        """get_similar_books does not return the query book itself."""
        model, _, _, book_idx = trained_svd
        book_id = list(book_idx.keys())[0]
        similar = model.get_similar_books(book_id, n_similar=10)
        assert book_id not in similar["book_id"].tolist()

    def test_similarity_scores_between_minus1_and_1(self, trained_svd):
        """Cosine similarity scores are in [-1, 1] range."""
        model, _, _, book_idx = trained_svd
        book_id = list(book_idx.keys())[0]
        similar = model.get_similar_books(book_id, n_similar=10)
        assert (similar["similarity_score"] >= -1.0).all()
        assert (similar["similarity_score"] <= 1.0).all()


# ── TF-IDF Recommender Tests ──────────────────────────────────────────────

class TestTFIDFRecommender:

    def test_model_is_fitted_after_training(self, trained_tfidf):
        assert trained_tfidf.is_fitted is True

    def test_book_vectors_correct_shape(self, trained_tfidf, lms_df):
        """Book vector matrix has correct dimensions."""
        n_books = lms_df["book_id"].nunique()
        assert trained_tfidf.book_vectors.shape[0] == n_books
        assert trained_tfidf.book_vectors.shape[1] == trained_tfidf.max_features

    def test_recommend_returns_dataframe(self, trained_tfidf, lms_df):
        patron_id = lms_df["patron_id"].iloc[0]
        recs = trained_tfidf.recommend(patron_id, lms_df, n_recommendations=5)
        assert isinstance(recs, pd.DataFrame)

    def test_recommend_has_correct_columns(self, trained_tfidf, lms_df):
        patron_id = lms_df["patron_id"].iloc[0]
        recs = trained_tfidf.recommend(patron_id, lms_df, n_recommendations=5)
        assert "book_id" in recs.columns
        assert "similarity_score" in recs.columns
        assert "genre" in recs.columns

    def test_similarity_scores_in_valid_range(self, trained_tfidf, lms_df):
        """Cosine similarity scores are in [-1, 1] range."""
        patron_id = lms_df["patron_id"].iloc[0]
        recs = trained_tfidf.recommend(patron_id, lms_df, n_recommendations=10)
        assert (recs["similarity_score"] >= -1.0).all()
        assert (recs["similarity_score"] <= 1.0).all()

    def test_excludes_already_seen_books(self, trained_tfidf, lms_df):
        """Recommendations exclude books patron has already borrowed."""
        patron_id = lms_df["patron_id"].iloc[0]
        seen = set(lms_df[lms_df["patron_id"] == patron_id]["book_id"].unique())
        recs = trained_tfidf.recommend(
            patron_id, lms_df, n_recommendations=20, exclude_seen=True
        )
        rec_books = set(recs["book_id"].tolist())
        assert len(seen & rec_books) == 0

    def test_similar_books_excludes_query_book(self, trained_tfidf, lms_df):
        """get_similar_books does not return the query book."""
        book_id = lms_df["book_id"].iloc[0]
        similar = trained_tfidf.get_similar_books(book_id, n_similar=10)
        assert book_id not in similar["book_id"].tolist()

    def test_explain_recommendation_returns_string(self, trained_tfidf, lms_df):
        """explain_recommendation returns a non-empty string."""
        patron_id = lms_df["patron_id"].iloc[0]
        recs = trained_tfidf.recommend(patron_id, lms_df, n_recommendations=5)
        if len(recs) > 0:
            explanation = trained_tfidf.explain_recommendation(
                patron_id, recs["book_id"].iloc[0], lms_df
            )
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_recommend_before_fitting_raises(self):
        """Calling recommend before fit raises RuntimeError."""
        from tfidf_model import TFIDFRecommender
        model = TFIDFRecommender()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.recommend("P00001", pd.DataFrame(), n_recommendations=5)


# ── Hybrid Engine Tests ───────────────────────────────────────────────────

class TestHybridEngine:

    @pytest.fixture(scope="class")
    def trained_engine(self, lms_df):
        """Pre-trained hybrid engine for testing."""
        from hybrid_engine import HybridRecommendationEngine
        engine = HybridRecommendationEngine(
            als_weight=0.35, svd_weight=0.25,
            tfidf_weight=0.20, bert_weight=0.20
        )
        engine.fit(lms_df, use_bert=False)  # Skip BERT for speed
        return engine, lms_df

    def test_engine_is_fitted(self, trained_engine):
        engine, _ = trained_engine
        assert engine.is_fitted is True

    def test_weights_sum_to_one(self, trained_engine):
        """Model weights sum to exactly 1.0."""
        engine, _ = trained_engine
        weights = engine.get_model_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_recommend_returns_dataframe(self, trained_engine):
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        recs = engine.recommend(patron_id, n_recommendations=10)
        assert isinstance(recs, pd.DataFrame)

    def test_recommend_has_correct_columns(self, trained_engine):
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        recs = engine.recommend(patron_id, n_recommendations=10)
        assert "book_id" in recs.columns
        assert "combined_score" in recs.columns
        assert "genre" in recs.columns

    def test_recommend_respects_n_recommendations(self, trained_engine):
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        for n in [5, 10]:
            recs = engine.recommend(patron_id, n_recommendations=n)
            assert len(recs) <= n

    def test_genre_diversity_cap_respected(self, trained_engine):
        """No genre appears more than diversity_cap times."""
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        recs = engine.recommend(patron_id, n_recommendations=20)
        if len(recs) > 0:
            genre_counts = recs["genre"].value_counts()
            assert (genre_counts <= engine.diversity_cap).all()

    def test_scores_ranked_descending(self, trained_engine):
        """Combined scores are in descending order."""
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        recs = engine.recommend(patron_id, n_recommendations=10)
        if len(recs) > 1:
            scores = recs["combined_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_explain_returns_string(self, trained_engine):
        """explain() returns a non-empty string."""
        engine, lms_df = trained_engine
        patron_id = lms_df["patron_id"].iloc[0]
        recs = engine.recommend(patron_id, n_recommendations=5)
        if len(recs) > 0:
            explanation = engine.explain(patron_id, recs["book_id"].iloc[0])
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_recommend_before_fitting_raises(self):
        """Calling recommend before fit raises RuntimeError."""
        from hybrid_engine import HybridRecommendationEngine
        engine = HybridRecommendationEngine()
        with pytest.raises(RuntimeError, match="must be fitted"):
            engine.recommend("P00001")