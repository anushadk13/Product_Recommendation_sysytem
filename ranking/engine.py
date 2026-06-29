"""
ranking/engine.py
──────────────────
The core ranking engine — the beating heart of the system.

Produces a final_score per (user, product) pair:

    final_score = (
        W_propensity  * purchase_propensity_score   # will this user buy?
      + W_inventory   * inventory_pressure_score    # do we need to move stock?
      + W_margin      * margin_score                # can we afford to discount?
      + W_trend       * trend_score                 # is this trending up?
      - W_diversity   * diversity_penalty           # avoid showing 10 similar products
    )

Weights are loaded from the A/B testing layer and are swappable per business goal.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data.feature_store import FeatureStore, ProductFeatures
from models.propensity import PropensityModel
from models.price_elasticity import PriceElasticityModel
from models.inventory_pressure import InventoryPressureModel
from models.demand_forecast import DemandForecastModel
from ranking.diversity import mmr_rerank
from ab_testing.feature_flags import WeightConfig, get_weight_config


@dataclass
class RankedProduct:
    """A single product in the ranked recommendation grid."""
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
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class RecommendationResult:
    user_id: str
    products: list[RankedProduct]
    weights_used: str
    latency_ms: float
    total_candidates: int


class RankingEngine:
    """
    Orchestrates all 4 models and produces a ranked product grid.
    Designed for sub-50ms p99 latency on pre-built feature store.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        propensity_model: PropensityModel,
        elasticity_model: PriceElasticityModel,
        inventory_model: InventoryPressureModel,
        demand_model: DemandForecastModel,
        products_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ):
        self.store = feature_store
        self.propensity = propensity_model
        self.elasticity = elasticity_model
        self.inventory = inventory_model
        self.demand = demand_model
        self.products_df = products_df
        self.events_df = events_df

        # Pre-compute inventory scores for all products (cached)
        print("[RankingEngine] Pre-computing inventory pressure scores...")
        self._inventory_scores: dict[str, float] = {}
        self._demand_scores: dict[str, float] = {}
        self._precompute_product_scores()
        print(f"[RankingEngine] Ready. {len(self._inventory_scores)} product scores cached.")

    def _precompute_product_scores(self) -> None:
        """Pre-compute static (non-user-specific) scores for all products."""
        pressures = self.inventory.predict_batch(self.products_df)
        for pid, score in zip(self.products_df["product_id"].values, pressures):
            self._inventory_scores[pid] = float(score)

        # Demand spike scores — use trend score as proxy when no recent events
        for _, row in self.products_df.iterrows():
            pid = row["product_id"]
            feat_vec = self.demand.get_product_forecast_features(
                pid, self.events_df, self.products_df
            )
            try:
                predicted = self.demand.predict_demand(feat_vec)
                spike = self.demand.demand_spike_score(pid, predicted)
            except Exception:
                spike = float(row.get("trend_score", 0.5))
            self._demand_scores[pid] = spike

    def recommend(
        self,
        user_id: str,
        top_k: int = 20,
        candidate_pool: int = 100,
        weight_preset: str = "balanced",
        apply_diversity: bool = True,
    ) -> RecommendationResult:
        """
        Generate a ranked product recommendation grid for a user.

        Args:
            user_id       : The user to personalize for
            top_k         : Number of products to return
            candidate_pool: Number of candidates to score before top_k selection
            weight_preset : A/B test weight config ("balanced", "lenskart", "dyson", "amazon")
            apply_diversity: Apply MMR diversity re-ranking

        Returns:
            RecommendationResult with ranked products
        """
        t_start = time.perf_counter()

        # Load weight config from A/B testing layer
        weights = get_weight_config(weight_preset)

        # Get user features
        user_feat = self.store.get_user_features(user_id)
        user_vec = self.store.user_to_vector(user_id)
        if user_vec is None:
            # Cold start: return trending products
            return self._cold_start_fallback(user_id, top_k, weights, t_start)

        # Get candidate products (random sample for efficiency)
        all_pids = self.store.get_all_product_ids()
        rng = np.random.default_rng(hash(user_id) % (2**32))
        candidates = rng.choice(all_pids, size=min(candidate_pool, len(all_pids)), replace=False).tolist()

        # Batch propensity scoring
        product_vecs = []
        valid_pids = []
        for pid in candidates:
            pvec = self.store.product_to_vector(pid)
            if pvec is not None:
                product_vecs.append(pvec)
                valid_pids.append(pid)

        propensity_scores = self.propensity.predict_batch(user_vec, product_vecs)

        # Build ranked list
        raw_scores: list[tuple[str, float]] = []
        for pid, prop_score in zip(valid_pids, propensity_scores):
            pf = self.store.get_product_features(pid)
            if pf is None:
                continue

            inventory_score = self._inventory_scores.get(pid, 0.5)
            margin = pf.margin_score
            trend = pf.trend_score
            demand_spike = self._demand_scores.get(pid, 0.5)

            # Core weighted formula
            final_score = (
                weights.propensity   * float(prop_score) +
                weights.inventory    * inventory_score +
                weights.margin       * margin +
                weights.trend        * trend +
                weights.demand_spike * demand_spike
            )

            raw_scores.append((pid, round(final_score, 6)))

        # Sort by score descending
        raw_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply MMR diversity re-ranking
        if apply_diversity:
            pf_lookup = {pid: self.store.get_product_features(pid) for pid, _ in raw_scores}
            pf_lookup = {k: v for k, v in pf_lookup.items() if v is not None}
            raw_scores = mmr_rerank(raw_scores, pf_lookup, top_k=top_k, lambda_diversity=weights.diversity)
        else:
            raw_scores = raw_scores[:top_k]

        # Build full result objects
        ranked_products = []
        for rank, (pid, score) in enumerate(raw_scores, 1):
            pf = self.store.get_product_features(pid)
            if pf is None:
                continue

            inv_score = self._inventory_scores.get(pid, 0.5)
            prop_score_val = float(propensity_scores[valid_pids.index(pid)]) if pid in valid_pids else 0.5
            demand_spike = self._demand_scores.get(pid, 0.5)

            # Get optimal discount suggestion from elasticity model
            try:
                disc_info = self.elasticity.optimal_discount(pid)
                opt_discount = disc_info["optimal_discount_pct"]
            except Exception:
                opt_discount = 0.0

            ranked_products.append(RankedProduct(
                product_id=pid,
                rank=rank,
                final_score=round(score, 4),
                propensity_score=round(prop_score_val, 4),
                inventory_pressure=round(inv_score, 4),
                margin_score=round(pf.margin_score, 4),
                trend_score=round(pf.trend_score, 4),
                demand_spike=round(demand_spike, 4),
                should_show_sale_badge=self.inventory.should_show_sale_badge(inv_score),
                urgency_label=self.inventory.urgency_label(inv_score),
                optimal_discount_pct=opt_discount,
                category=pf.category,
                brand=pf.brand,
                current_price=round(pf.current_price, 2),
                base_price=round(pf.base_price, 2),
                avg_rating=pf.avg_rating,
                score_breakdown={
                    "propensity":     round(weights.propensity * prop_score_val, 4),
                    "inventory":      round(weights.inventory * inv_score, 4),
                    "margin":         round(weights.margin * pf.margin_score, 4),
                    "trend":          round(weights.trend * pf.trend_score, 4),
                    "demand_spike":   round(weights.demand_spike * demand_spike, 4),
                },
            ))

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return RecommendationResult(
            user_id=user_id,
            products=ranked_products,
            weights_used=weight_preset,
            latency_ms=latency_ms,
            total_candidates=len(valid_pids),
        )

    def score_pair(self, user_id: str, product_id: str, weight_preset: str = "balanced") -> dict:
        """
        Score a single (user, product) pair. Used by /score API endpoint.
        """
        weights = get_weight_config(weight_preset)
        user_vec = self.store.user_to_vector(user_id)
        prod_vec = self.store.product_to_vector(product_id)

        if user_vec is None or prod_vec is None:
            return {"error": "User or product not found"}

        pf = self.store.get_product_features(product_id)
        prop_score = self.propensity.predict(user_vec, prod_vec)
        inv_score = self._inventory_scores.get(product_id, 0.5)
        demand_spike = self._demand_scores.get(product_id, 0.5)

        try:
            disc_info = self.elasticity.optimal_discount(product_id)
        except Exception:
            disc_info = {"optimal_discount_pct": 0.0, "expected_uplift": 0.0}

        final_score = (
            weights.propensity   * prop_score +
            weights.inventory    * inv_score +
            weights.margin       * (pf.margin_score if pf else 0.5) +
            weights.trend        * (pf.trend_score if pf else 0.5) +
            weights.demand_spike * demand_spike
        )

        return {
            "user_id": user_id,
            "product_id": product_id,
            "final_score": round(final_score, 4),
            "propensity_score": round(prop_score, 4),
            "inventory_pressure": round(inv_score, 4),
            "margin_score": round(pf.margin_score if pf else 0.5, 4),
            "trend_score": round(pf.trend_score if pf else 0.5, 4),
            "demand_spike": round(demand_spike, 4),
            "should_show_sale_badge": self.inventory.should_show_sale_badge(inv_score),
            "urgency_label": self.inventory.urgency_label(inv_score),
            "optimal_discount_pct": disc_info.get("optimal_discount_pct", 0.0),
            "expected_uplift": disc_info.get("expected_uplift", 0.0),
            "weights_used": weight_preset,
        }

    def _cold_start_fallback(
        self,
        user_id: str,
        top_k: int,
        weights: "WeightConfig",
        t_start: float,
    ) -> RecommendationResult:
        """For unknown users: return top trending + high-margin products."""
        all_pids = self.store.get_all_product_ids()
        scores = []
        for pid in all_pids[:200]:
            pf = self.store.get_product_features(pid)
            if pf is None:
                continue
            score = 0.5 * pf.trend_score + 0.3 * pf.margin_score + 0.2 * pf.avg_rating / 5.0
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]

        ranked = []
        for rank, (pid, score) in enumerate(scores, 1):
            pf = self.store.get_product_features(pid)
            inv = self._inventory_scores.get(pid, 0.5)
            ranked.append(RankedProduct(
                product_id=pid, rank=rank, final_score=round(score, 4),
                propensity_score=0.5, inventory_pressure=round(inv, 4),
                margin_score=pf.margin_score, trend_score=pf.trend_score,
                demand_spike=self._demand_scores.get(pid, 0.5),
                should_show_sale_badge=self.inventory.should_show_sale_badge(inv),
                urgency_label=self.inventory.urgency_label(inv),
                optimal_discount_pct=0.0, category=pf.category,
                brand=pf.brand, current_price=pf.current_price,
                base_price=pf.base_price, avg_rating=pf.avg_rating,
                score_breakdown={},
            ))

        latency = round((time.perf_counter() - t_start) * 1000, 2)
        return RecommendationResult(
            user_id=user_id, products=ranked, weights_used="cold_start",
            latency_ms=latency, total_candidates=len(all_pids),
        )
