"""
api/schemas.py
--------------
Pydantic request and response schemas for the recommendation API.

Why Pydantic schemas?
- Automatic request validation — bad inputs are rejected before
  they reach model code
- Auto-generated API documentation (FastAPI uses these for /docs)
- Clear contract between API consumers and the recommendation engine
- Type safety — catches bugs at the boundary, not deep in model code
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# ── Request Schemas ───────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """
    Request body for POST /recommend endpoint.

    Example:
        {
            "patron_id": "P00001",
            "n_recommendations": 10,
            "exclude_seen": true,
            "diversity_cap": 3
        }
    """
    patron_id: str = Field(
        ...,
        description="Unique patron identifier",
        example="P00001",
        min_length=1,
        max_length=20
    )
    n_recommendations: int = Field(
        default=10,
        description="Number of recommendations to return",
        ge=1,    # greater than or equal to 1
        le=50    # less than or equal to 50
    )
    exclude_seen: bool = Field(
        default=True,
        description="Whether to exclude books the patron has already borrowed"
    )
    diversity_cap: Optional[int] = Field(
        default=3,
        description="Maximum recommendations from any single genre",
        ge=1,
        le=10
    )

    @validator("patron_id")
    def patron_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("patron_id cannot be empty or whitespace")
        return v.strip()


class SimilarBooksRequest(BaseModel):
    """
    Request body for POST /similar-books endpoint.

    Example:
        {
            "book_id": "B00001",
            "n_similar": 10,
            "method": "bert"
        }
    """
    book_id: str = Field(
        ...,
        description="Book ID to find similar books for",
        example="B00001"
    )
    n_similar: int = Field(
        default=10,
        description="Number of similar books to return",
        ge=1,
        le=50
    )
    method: str = Field(
        default="tfidf",
        description="Similarity method: 'tfidf' or 'bert'",
        example="tfidf"
    )

    @validator("method")
    def method_must_be_valid(cls, v):
        valid_methods = {"tfidf", "bert"}
        if v not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        return v


class ExplainRequest(BaseModel):
    """
    Request body for POST /explain endpoint.

    Example:
        {
            "patron_id": "P00001",
            "book_id": "B00001"
        }
    """
    patron_id: str = Field(..., description="Patron ID", example="P00001")
    book_id: str = Field(..., description="Book ID to explain", example="B00001")


# ── Response Schemas ──────────────────────────────────────────────────────

class BookRecommendation(BaseModel):
    """A single book recommendation with score and metadata."""
    book_id: str = Field(..., description="Unique book identifier")
    combined_score: float = Field(
        ...,
        description="Hybrid recommendation score (0-1, higher is better)"
    )
    genre: str = Field(..., description="Book genre")
    rank: int = Field(..., description="Recommendation rank (1 = best)")


class RecommendationResponse(BaseModel):
    """
    Response from POST /recommend endpoint.

    Example:
        {
            "patron_id": "P00001",
            "recommendations": [...],
            "model_weights": {...},
            "generated_at": "2026-09-05T10:30:00",
            "inference_time_ms": 45.2
        }
    """
    patron_id: str
    recommendations: List[BookRecommendation]
    model_weights: dict = Field(
        ...,
        description="Weights used by each model in the hybrid engine"
    )
    generated_at: datetime
    inference_time_ms: float = Field(
        ...,
        description="Recommendation inference time in milliseconds"
    )
    n_models_used: int = Field(
        ...,
        description="Number of models that contributed to recommendations"
    )


class SimilarBookItem(BaseModel):
    """A single similar book result."""
    book_id: str
    similarity_score: float
    genre: str
    rank: int


class SimilarBooksResponse(BaseModel):
    """Response from POST /similar-books endpoint."""
    book_id: str
    similar_books: List[SimilarBookItem]
    method: str
    generated_at: datetime


class ExplainResponse(BaseModel):
    """Response from POST /explain endpoint."""
    patron_id: str
    book_id: str
    explanation: str
    generated_at: datetime


class HealthResponse(BaseModel):
    """Response from GET /health endpoint."""
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    model_loaded: bool
    n_patrons_in_model: int
    n_books_in_model: int
    uptime_seconds: float
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Response from GET /model/info endpoint."""
    model_type: str = "HybridRecommendationEngine"
    components: List[str]
    weights: dict
    n_patrons: int
    n_books: int
    is_fitted: bool