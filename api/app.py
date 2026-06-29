"""
api/app.py
───────────
FastAPI application — the real-time serving layer.

Endpoints:
  POST /recommend          → ranked product grid for a user
  POST /score              → score a single (user, product) pair
  GET  /sale-badge/{id}    → should this product show a sale badge?
  GET  /weight-presets     → list all A/B weight configurations
  GET  /health             → health check

Designed for sub-50ms p99 latency using pre-built feature store.
"""

from __future__ import annotations

import sys
import os
# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.schemas import (
    RecommendRequest, RecommendResponse, RankedProductResponse,
    ScoreRequest, ScoreResponse,
    SaleBadgeResponse,
    HealthResponse, WeightPresetResponse,
)
from ab_testing.feature_flags import list_presets, assign_ab_variant


# ─── App state (populated on startup) ────────────────────────────────────────

_engine = None
_feature_store = None
_inventory_model = None
_elasticity_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and build feature store at startup."""
    global _engine, _feature_store, _inventory_model, _elasticity_model
    print("\n[API] 🚀 Starting up — loading models...")

    from file import build_system
    system = build_system(verbose=False)

    _engine = system["engine"]
    _feature_store = system["feature_store"]
    _inventory_model = system["inventory_model"]
    _elasticity_model = system["elasticity_model"]

    print("[API] ✅ All models loaded — server ready\n")
    yield
    print("[API] Shutting down...")


app = FastAPI(
    title="Personalized Merchandising API",
    description=(
        "Real-time personalized product recommendation engine. "
        "Combines 4 ML models (propensity, price elasticity, inventory pressure, "
        "demand forecast) into a weighted ranking formula."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _engine is not None else "initializing",
        models_loaded=_engine is not None,
        n_users=len(_feature_store.get_all_user_ids()) if _feature_store else 0,
        n_products=len(_feature_store.get_all_product_ids()) if _feature_store else 0,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(req: RecommendRequest):
    """
    Generate a ranked product grid for a user.

    - Uses all 4 ML models for scoring
    - Applies MMR diversity re-ranking (optional)
    - Weights can be switched per A/B preset

    **Latency target**: < 50ms p99
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    result = _engine.recommend(
        user_id=req.user_id,
        top_k=req.top_k,
        candidate_pool=req.candidate_pool,
        weight_preset=req.weight_preset,
        apply_diversity=req.apply_diversity,
    )

    products_out = [
        RankedProductResponse(
            product_id=p.product_id,
            rank=p.rank,
            final_score=p.final_score,
            propensity_score=p.propensity_score,
            inventory_pressure=p.inventory_pressure,
            margin_score=p.margin_score,
            trend_score=p.trend_score,
            demand_spike=p.demand_spike,
            should_show_sale_badge=p.should_show_sale_badge,
            urgency_label=p.urgency_label,
            optimal_discount_pct=p.optimal_discount_pct,
            category=p.category,
            brand=p.brand,
            current_price=p.current_price,
            base_price=p.base_price,
            avg_rating=p.avg_rating,
            score_breakdown=p.score_breakdown,
        )
        for p in result.products
    ]

    return RecommendResponse(
        user_id=result.user_id,
        products=products_out,
        weights_used=result.weights_used,
        latency_ms=result.latency_ms,
        total_candidates=result.total_candidates,
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_pair(req: ScoreRequest):
    """
    Score a single (user, product) pair across all dimensions.

    Returns the full score breakdown including:
    - Purchase propensity
    - Inventory pressure
    - Price elasticity / optimal discount
    - Demand spike
    - Sale badge recommendation
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    result = _engine.score_pair(
        user_id=req.user_id,
        product_id=req.product_id,
        weight_preset=req.weight_preset,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return ScoreResponse(**result)


@app.get("/sale-badge/{product_id}", response_model=SaleBadgeResponse, tags=["Merchandising"])
async def get_sale_badge(product_id: str):
    """
    Determine if a product should show a sale badge on the storefront.

    Based on:
    - Inventory pressure score (XGBoost model)
    - Urgency label (Low / Medium / High / Critical)
    - Optimal discount recommendation from elasticity model
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    inv_score = _engine._inventory_scores.get(product_id)
    if inv_score is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    try:
        disc_info = _inventory_model.optimal_discount(product_id) if hasattr(_inventory_model, 'optimal_discount') else {"optimal_discount_pct": 0.0, "expected_uplift": 0.0}
    except Exception:
        disc_info = {"optimal_discount_pct": 0.0, "expected_uplift": 0.0}

    try:
        disc_info = _elasticity_model.optimal_discount(product_id)
    except Exception:
        pass

    return SaleBadgeResponse(
        product_id=product_id,
        should_show_sale_badge=_inventory_model.should_show_sale_badge(inv_score),
        inventory_pressure_score=round(inv_score, 4),
        urgency_label=_inventory_model.urgency_label(inv_score),
        optimal_discount_pct=disc_info.get("optimal_discount_pct", 0.0),
        expected_uplift=disc_info.get("expected_uplift", 0.0),
    )


@app.get("/weight-presets", response_model=WeightPresetResponse, tags=["Configuration"])
async def get_weight_presets():
    """List all available A/B test weight presets and their formulas."""
    return WeightPresetResponse(presets=list_presets())


@app.get("/ab-variant/{user_id}", tags=["A/B Testing"])
async def get_ab_variant(user_id: str, experiment: str = "ranking_weights"):
    """Get the deterministic A/B test variant for a user."""
    variant = assign_ab_variant(user_id, experiment)
    return {
        "user_id": user_id,
        "experiment": experiment,
        "variant": variant,
        "description": f"User {user_id} is assigned to the '{variant}' weight preset",
    }


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
