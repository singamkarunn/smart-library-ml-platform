"""
tests/test_api.py
------------------
Integration tests for the FastAPI recommendation API.

Uses FastAPI's TestClient to test endpoints without running
a real server — requests go directly to the app in-process.

Tests verify:
- All endpoints return correct status codes
- Response schemas match Pydantic models
- Error cases return appropriate HTTP status codes
- Latency is within acceptable bounds

Run: python -m pytest tests/test_api.py -v
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "collaborative"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "content_based"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from fastapi.testclient import TestClient


# ── App Setup ─────────────────────────────────────────────────────────────
# We need to initialize the app with a fitted model before testing.
# This fixture runs once for the entire test session.

@pytest.fixture(scope="session")
def client():
    """
    Creates a TestClient with a fully initialized app.

    Patches the global engine and lms_df so tests don't need
    a running server or real database.
    """
    import main as api_main
    from lms_connector import load_lms_data
    from hybrid_engine import HybridRecommendationEngine

    # Load small dataset for fast testing
    lms_df = load_lms_data(source="synthetic", n_transactions=1000, seed=42)

    # Fit lightweight engine
    engine = HybridRecommendationEngine(
        als_weight=0.35, svd_weight=0.25,
        tfidf_weight=0.20, bert_weight=0.20
    )
    engine.fit(lms_df, use_bert=False)

    # Inject into app globals
    api_main.engine = engine
    api_main.lms_df = lms_df
    api_main.startup_time = time.time()

    with TestClient(api_main.app) as c:
        yield c, lms_df


# ── Health Endpoint Tests ─────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        c, _ = client
        response = c.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_model_loaded_true(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_has_patron_count(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["n_patrons_in_model"] > 0

    def test_health_has_book_count(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["n_books_in_model"] > 0

    def test_health_has_uptime(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["uptime_seconds"] >= 0

    def test_health_has_version(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert "version" in data


# ── Model Info Endpoint Tests ─────────────────────────────────────────────

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        c, _ = client
        response = c.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_components(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert "components" in data
        assert len(data["components"]) > 0

    def test_model_info_has_weights(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert "weights" in data
        weights = data["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_model_info_is_fitted(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert data["is_fitted"] is True

    def test_model_info_patron_count_positive(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert data["n_patrons"] > 0


# ── Recommend Endpoint Tests ──────────────────────────────────────────────

class TestRecommendEndpoint:

    def test_recommend_returns_200_for_valid_patron(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        response = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        })
        assert response.status_code == 200

    def test_recommend_returns_correct_patron_id(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        assert data["patron_id"] == patron_id

    def test_recommend_returns_list_of_recommendations(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_recommend_respects_n_recommendations(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        for n in [3, 5, 10]:
            data = c.post("/recommend", json={
                "patron_id": patron_id,
                "n_recommendations": n
            }).json()
            assert len(data["recommendations"]) <= n

    def test_recommend_response_has_inference_time(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0

    def test_recommend_response_has_model_weights(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        assert "model_weights" in data

    def test_recommend_book_has_required_fields(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        if data["recommendations"]:
            book = data["recommendations"][0]
            assert "book_id" in book
            assert "combined_score" in book
            assert "genre" in book
            assert "rank" in book

    def test_recommend_ranks_start_at_1(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        data = c.post("/recommend", json={
            "patron_id": patron_id,
            "n_recommendations": 5
        }).json()
        if data["recommendations"]:
            assert data["recommendations"][0]["rank"] == 1

    def test_recommend_unknown_patron_returns_404(self, client):
        c, _ = client
        response = c.post("/recommend", json={
            "patron_id": "P99999",
            "n_recommendations": 5
        })
        assert response.status_code == 404

    def test_recommend_empty_patron_id_returns_422(self, client):
        c, _ = client
        response = c.post("/recommend", json={
            "patron_id": "",
            "n_recommendations": 5
        })
        assert response.status_code == 422

    def test_recommend_invalid_n_returns_422(self, client):
        """n_recommendations must be >= 1."""
        c, lms_df = client
        response = c.post("/recommend", json={
            "patron_id": lms_df["patron_id"].iloc[0],
            "n_recommendations": 0
        })
        assert response.status_code == 422

    def test_recommend_n_too_large_returns_422(self, client):
        """n_recommendations must be <= 50."""
        c, lms_df = client
        response = c.post("/recommend", json={
            "patron_id": lms_df["patron_id"].iloc[0],
            "n_recommendations": 100
        })
        assert response.status_code == 422

    def test_recommend_inference_under_300ms(self, client):
        """P95 inference latency must be under 300ms SLA."""
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        latencies = []
        for _ in range(5):
            data = c.post("/recommend", json={
                "patron_id": patron_id,
                "n_recommendations": 10
            }).json()
            latencies.append(data["inference_time_ms"])
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 300, f"P95 latency {p95:.1f}ms exceeds 300ms SLA"


# ── Similar Books Endpoint Tests ──────────────────────────────────────────

class TestSimilarBooksEndpoint:

    def test_similar_books_returns_200(self, client):
        c, lms_df = client
        book_id = lms_df["book_id"].iloc[0]
        response = c.post("/similar-books", json={
            "book_id": book_id,
            "n_similar": 5,
            "method": "tfidf"
        })
        assert response.status_code == 200

    def test_similar_books_returns_list(self, client):
        c, lms_df = client
        book_id = lms_df["book_id"].iloc[0]
        data = c.post("/similar-books", json={
            "book_id": book_id,
            "n_similar": 5,
            "method": "tfidf"
        }).json()
        assert "similar_books" in data
        assert isinstance(data["similar_books"], list)

    def test_similar_books_respects_n_similar(self, client):
        c, lms_df = client
        book_id = lms_df["book_id"].iloc[0]
        data = c.post("/similar-books", json={
            "book_id": book_id,
            "n_similar": 5,
            "method": "tfidf"
        }).json()
        assert len(data["similar_books"]) <= 5

    def test_similar_books_invalid_method_returns_422(self, client):
        c, lms_df = client
        response = c.post("/similar-books", json={
            "book_id": lms_df["book_id"].iloc[0],
            "n_similar": 5,
            "method": "invalid_method"
        })
        assert response.status_code == 422

    def test_similar_books_unknown_book_returns_404(self, client):
        c, _ = client
        response = c.post("/similar-books", json={
            "book_id": "UNKNOWN_BOOK_XYZ_999",
            "n_similar": 5,
            "method": "tfidf"
        })
        assert response.status_code == 404


# ── Explain Endpoint Tests ────────────────────────────────────────────────

class TestExplainEndpoint:

    def test_explain_returns_200(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        book_id = lms_df["book_id"].iloc[0]
        response = c.post("/explain", json={
            "patron_id": patron_id,
            "book_id": book_id
        })
        assert response.status_code == 200

    def test_explain_returns_string_explanation(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        book_id = lms_df["book_id"].iloc[0]
        data = c.post("/explain", json={
            "patron_id": patron_id,
            "book_id": book_id
        }).json()
        assert "explanation" in data
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 0

    def test_explain_response_has_patron_and_book_ids(self, client):
        c, lms_df = client
        patron_id = lms_df["patron_id"].iloc[0]
        book_id = lms_df["book_id"].iloc[0]
        data = c.post("/explain", json={
            "patron_id": patron_id,
            "book_id": book_id
        }).json()
        assert data["patron_id"] == patron_id
        assert data["book_id"] == book_id

    def test_explain_has_generated_at_timestamp(self, client):
        c, lms_df = client
        data = c.post("/explain", json={
            "patron_id": lms_df["patron_id"].iloc[0],
            "book_id": lms_df["book_id"].iloc[0]
        }).json()
        assert "generated_at" in data


# ── Run all tests ─────────────────────────────────────────────────────────
# python -m pytest tests/test_api.py -v --tb=short
