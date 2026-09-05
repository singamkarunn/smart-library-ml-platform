"""
api/main.py
-----------
FastAPI REST API serving the hybrid recommendation engine.

Endpoints:
    POST /recommend          — Get top-N book recommendations for a patron
    POST /similar-books      — Find books similar to a given book
    POST /explain            — Explain why a book was recommended
    GET  /health             — Service health check
    GET  /model/info         — Model metadata and configuration
    GET  /docs               — Auto-generated API documentation (Swagger UI)

Why FastAPI?
- Automatic request validation via Pydantic schemas
- Auto-generated /docs endpoint (Swagger UI) — zero extra work
- Async support for high-throughput inference
- Production-ready: used at Uber, Microsoft, Netflix for ML serving
- Type hints throughout = self-documenting code

Performance:
- Model loaded once at startup into memory
- Each request only runs inference — no reloading
- Target: <300ms P95 latency for recommendation requests
"""

import time
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(API_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "ingestion"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "collaborative"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "content_based"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models"))

from schemas import (
    RecommendationRequest, RecommendationResponse, BookRecommendation,
    SimilarBooksRequest, SimilarBooksResponse, SimilarBookItem,
    ExplainRequest, ExplainResponse,
    HealthResponse, ModelInfoResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global State ──────────────────────────────────────────────────────────
# Model is loaded once at startup and reused for every request.
# This is critical for performance — loading the model per-request
# would add 30-60 seconds of latency.
engine = None
lms_df = None
startup_time = None


# ── App Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    On startup: loads and fits the hybrid recommendation engine.
    On shutdown: cleans up resources.

    Using lifespan instead of @app.on_event is the modern FastAPI
    pattern — it's more explicit about resource lifecycle.
    """
    global engine, lms_df, startup_time
    startup_time = time.time()

    logger.info("Starting Smart Library Recommendation API...")
    logger.info("Loading data and fitting models...")

    try:
        from lms_connector import load_lms_data
        from hybrid_engine import HybridRecommendationEngine

        # Load training data
        data_path = os.path.join(ROOT_DIR, "data", "lms_raw.parquet")
        if os.path.exists(data_path):
            import pandas as pd
            lms_df = pd.read_parquet(data_path)
            logger.info(f"Loaded {len(lms_df)} records from ETL output")
        else:
            lms_df = load_lms_data(source="synthetic", n_transactions=5000)
            logger.info(f"Generated {len(lms_df)} synthetic records")

        # Fit hybrid engine
        engine = HybridRecommendationEngine(
            als_weight=0.35,
            svd_weight=0.25,
            tfidf_weight=0.20,
            bert_weight=0.20
        )
        engine.fit(lms_df, use_bert=True)

        logger.info("Hybrid engine ready — API accepting requests")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    yield  # API runs here

    logger.info("Shutting down Smart Library Recommendation API...")


# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Library Recommendation API",
    description="""
## Smart Library ML Platform — Recommendation API

Serving personalized book recommendations via a hybrid ML engine combining:
- **ALS** collaborative filtering (weight: 0.35)
- **SVD** collaborative filtering (weight: 0.25)
- **TF-IDF** content-based filtering (weight: 0.20)
- **BERT** semantic content filtering (weight: 0.20)

### Key Features
- Genre diversity constraints (max 3 books per genre)
- Excludes already-borrowed books by default
- Explainable recommendations
- <300ms P95 inference latency
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS Middleware ───────────────────────────────────────────────────────
# Allows the Streamlit dashboard to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds X-Process-Time header to every response for latency monitoring."""
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Service health check.

    Returns model status, patron/book counts, and uptime.
    Used by Prometheus and load balancers to verify service health.
    """
    if engine is None or not engine.is_fitted:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            n_patrons_in_model=0,
            n_books_in_model=0,
            uptime_seconds=0.0
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        n_patrons_in_model=len(engine.patron_index) if engine.patron_index else 0,
        n_books_in_model=len(engine.book_index) if engine.book_index else 0,
        uptime_seconds=time.time() - startup_time if startup_time else 0.0
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """
    Returns metadata about the loaded recommendation model.

    Includes component models, weights, and training data size.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    components = ["ALS", "SVD", "TF-IDF"]
    if engine.bert_model is not None:
        components.append("BERT")

    return ModelInfoResponse(
        components=components,
        weights=engine.get_model_weights(),
        n_patrons=len(engine.patron_index) if engine.patron_index else 0,
        n_books=len(engine.book_index) if engine.book_index else 0,
        is_fitted=engine.is_fitted
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized book recommendations for a patron.

    Runs the full hybrid pipeline:
    1. Gets candidates from ALS, SVD, TF-IDF, and BERT
    2. Normalizes and fuses scores with configurable weights
    3. Applies genre diversity constraints
    4. Returns ranked recommendations with scores

    **Example request:**
```json
    {
        "patron_id": "P00001",
        "n_recommendations": 10,
        "exclude_seen": true
    }
```
    """
    if engine is None or not engine.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        recs_df = engine.recommend(
            patron_id=request.patron_id,
            n_recommendations=request.n_recommendations
        )
    except Exception as e:
        logger.error(f"Recommendation failed for {request.patron_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {str(e)}"
        )

    inference_ms = (time.time() - start_time) * 1000

    if len(recs_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for patron {request.patron_id}. "
                   f"Patron may not exist in training data."
        )

    recommendations = [
        BookRecommendation(
            book_id=row["book_id"],
            combined_score=float(row["combined_score"]),
            genre=str(row.get("genre", "Unknown")),
            rank=idx + 1
        )
        for idx, row in recs_df.iterrows()
    ]

    models_used = []
    if engine.als_model: models_used.append("als")
    if engine.svd_model: models_used.append("svd")
    if engine.tfidf_model: models_used.append("tfidf")
    if engine.bert_model: models_used.append("bert")

    logger.info(
        f"Recommendations served for {request.patron_id} — "
        f"{len(recommendations)} books in {inference_ms:.1f}ms"
    )

    return RecommendationResponse(
        patron_id=request.patron_id,
        recommendations=recommendations,
        model_weights=engine.get_model_weights(),
        generated_at=datetime.now(),
        inference_time_ms=round(inference_ms, 2),
        n_models_used=len(models_used)
    )


@app.post("/similar-books",
          response_model=SimilarBooksResponse,
          tags=["Recommendations"])
async def get_similar_books(request: SimilarBooksRequest):
    """
    Find books similar to a given book.

    Uses TF-IDF or BERT embeddings to find books with similar
    content — useful for "readers also enjoyed" style features.
    """
    if engine is None or not engine.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if request.method == "bert" and engine.bert_model:
            similar_df = engine.bert_model.get_similar_books(
                request.book_id, n_similar=request.n_similar
            )
        else:
            similar_df = engine.tfidf_model.get_similar_books(
                request.book_id, n_similar=request.n_similar
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if len(similar_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Book {request.book_id} not found in catalog"
        )

    similar_books = [
        SimilarBookItem(
            book_id=row["book_id"],
            similarity_score=float(row["similarity_score"]),
            genre=str(row.get("genre", "Unknown")),
            rank=idx + 1
        )
        for idx, row in similar_df.iterrows()
    ]

    return SimilarBooksResponse(
        book_id=request.book_id,
        similar_books=similar_books,
        method=request.method,
        generated_at=datetime.now()
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Recommendations"])
async def explain_recommendation(request: ExplainRequest):
    """
    Explain why a book was recommended to a patron.

    Returns a human-readable explanation identifying the key signals
    that drove the recommendation — genre match, collaborative signal,
    or semantic similarity.
    """
    if engine is None or not engine.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        explanation = engine.explain(request.patron_id, request.book_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExplainResponse(
        patron_id=request.patron_id,
        book_id=request.book_id,
        explanation=explanation,
        generated_at=datetime.now()
    )


# ── Startup test ──────────────────────────────────────────────────────────
# Run: python api/main.py
# Then visit: http://localhost:8000/docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )