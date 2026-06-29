"""
api/schemas.py
───────────────
Pydantic v2 request/response schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ─── Request schemas ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="The user to generate recommendations for", example="U00001")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of products to return")
    candidate_pool: int = Field(default=100, ge=10, le=500, description="Number of candidates to score")
    weight_preset: str = Field(default="balanced", description="A/B weight preset: balanced|lenskart|dyson|amazon|clearance")
    apply_diversity: bool = Field(default=True, description="Apply MMR diversity re-ranking")


class ScoreRequest(BaseModel):
    user_id: str = Field(..., description="User ID", example="U00001")
    product_id: str = Field(..., description="Product ID", example="P00042")
    weight_preset: str = Field(default="balanced", description="A/B weight preset")


class SaleBadgeRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to check for sale badge")


# ─── Response schemas ─────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    propensity: float
    inventory: float
    margin: float
    trend: float
    demand_spike: float


class RankedProductResponse(BaseModel):
    product_id: str
    rank: int
    final_score: float
    propensity_score: float
    inventory_pressure: float
    margin_score: float
    trend_score: float
    demand_spike: float
    should_show_sale_badge: bool
    urgency_label: str
    optimal_discount_pct: float
    category: str
    brand: str
    current_price: float
    base_price: float
    avg_rating: float
    score_breakdown: Optional[dict] = None


class RecommendResponse(BaseModel):
    user_id: str
    products: list[RankedProductResponse]
    weights_used: str
    latency_ms: float
    total_candidates: int
    status: str = "ok"


class ScoreResponse(BaseModel):
    user_id: str
    product_id: str
    final_score: float
    propensity_score: float
    inventory_pressure: float
    margin_score: float
    trend_score: float
    demand_spike: float
    should_show_sale_badge: bool
    urgency_label: str
    optimal_discount_pct: float
    expected_uplift: float
    weights_used: str
    status: str = "ok"


class SaleBadgeResponse(BaseModel):
    product_id: str
    should_show_sale_badge: bool
    inventory_pressure_score: float
    urgency_label: str
    optimal_discount_pct: float
    expected_uplift: float
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    n_users: int
    n_products: int
    version: str = "1.0.0"


class WeightPresetResponse(BaseModel):
    presets: list[dict]
