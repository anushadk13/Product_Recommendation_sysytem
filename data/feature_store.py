"""
data/feature_store.py
──────────────────────
In-memory feature store for low-latency feature serving.

Mimics a production Redis / Feast feature store:
  - Pre-computes user feature vectors at startup
  - Pre-computes product feature vectors at startup
  - Serves features in O(1) for ranking at inference time

Usage:
    store = FeatureStore()
    store.build(users_df, products_df, events_df)
    user_feats    = store.get_user_features("U00001")
    product_feats = store.get_product_features("P00042")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


# ─── Feature Vectors ──────────────────────────────────────────────────────────

@dataclass
class UserFeatures:
    user_id: str
    recency_days: float
    purchase_freq_90d: float
    avg_order_value: float
    lifetime_spend: float
    price_sensitivity: float
    sessions_7d: float
    cart_abandon_rate: float
    account_age_days: float
    segment_encoded: float          # 0=New, 1=Lapsed, 2=Regular, 3=VIP
    category_affinities: dict       # {category: affinity_score}
    # Derived at build time from events
    purchase_count_30d: int = 0
    last_category_viewed: str = ""
    avg_discount_accepted: float = 0.0


@dataclass
class ProductFeatures:
    product_id: str
    category: str
    brand: str
    base_price: float
    current_price: float
    cost_price: float
    current_stock: int
    sell_through_rate: float
    days_on_shelf: int
    reorder_threshold: int
    avg_rating: float
    review_count: int
    trend_score: float
    margin_score: float
    is_new_launch: int
    style_cluster: int
    seasonal_relevance: float


# ─── Feature Store ─────────────────────────────────────────────────────────────

SEGMENT_MAP = {"New": 0, "Lapsed": 1, "Regular": 2, "VIP": 3}
CATEGORIES = ["Eyewear", "Electronics", "Home", "Fashion", "Beauty", "Sports"]


class FeatureStore:
    """
    In-memory feature store.  Build once at startup, serve many times.
    In production this would be backed by Redis with a TTL.
    """

    def __init__(self):
        self._users: dict[str, UserFeatures] = {}
        self._products: dict[str, ProductFeatures] = {}
        self._is_built = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(
        self,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> None:
        """Pre-compute and cache all feature vectors."""
        print("[FeatureStore] Building user features...")
        self._build_user_features(users_df, events_df)

        print("[FeatureStore] Building product features...")
        self._build_product_features(products_df)

        self._is_built = True
        print(f"[FeatureStore] Done. {len(self._users)} users, {len(self._products)} products cached.")

    def _build_user_features(self, users_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
        # Compute 30-day purchase counts from events
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True)
        recent_purchases = (
            events_df[
                (events_df["event_type"] == "purchase") &
                (events_df["timestamp"] >= cutoff)
            ]
            .groupby("user_id")
            .size()
            .to_dict()
        )

        # Last category viewed per user
        last_events = (
            events_df[events_df["event_type"] == "view"]
            .sort_values("timestamp")
            .groupby("user_id")["product_id"]
            .last()
            .to_dict()
        )

        # Avg discount accepted per user (price paid vs base price)
        # Simplified: track avg price_paid / events price ratio
        avg_discount = (
            events_df[events_df["event_type"] == "purchase"]
            .groupby("user_id")["price_paid"]
            .mean()
            .to_dict()
        )

        for _, row in users_df.iterrows():
            uid = row["user_id"]
            affinities = {cat: row.get(f"affinity_{cat.lower()}", 0.0) for cat in CATEGORIES}

            self._users[uid] = UserFeatures(
                user_id=uid,
                recency_days=float(row["recency_days"]),
                purchase_freq_90d=float(row["purchase_freq_90d"]),
                avg_order_value=float(row["avg_order_value"]),
                lifetime_spend=float(row["lifetime_spend"]),
                price_sensitivity=float(row["price_sensitivity"]),
                sessions_7d=float(row["sessions_7d"]),
                cart_abandon_rate=float(row["cart_abandon_rate"]),
                account_age_days=float(row["account_age_days"]),
                segment_encoded=float(SEGMENT_MAP.get(row.get("segment", "New"), 0)),
                category_affinities=affinities,
                purchase_count_30d=int(recent_purchases.get(uid, 0)),
                last_category_viewed="",
                avg_discount_accepted=float(avg_discount.get(uid, row["avg_order_value"])),
            )

    def _build_product_features(self, products_df: pd.DataFrame) -> None:
        for _, row in products_df.iterrows():
            pid = row["product_id"]
            self._products[pid] = ProductFeatures(
                product_id=pid,
                category=row["category"],
                brand=row["brand"],
                base_price=float(row["base_price"]),
                current_price=float(row["current_price"]),
                cost_price=float(row["cost_price"]),
                current_stock=int(row["current_stock"]),
                sell_through_rate=float(row["sell_through_rate"]),
                days_on_shelf=int(row["days_on_shelf"]),
                reorder_threshold=int(row["reorder_threshold"]),
                avg_rating=float(row["avg_rating"]),
                review_count=int(row["review_count"]),
                trend_score=float(row["trend_score"]),
                margin_score=float(row["margin_score"]),
                is_new_launch=int(row["is_new_launch"]),
                style_cluster=int(row["style_cluster"]),
                seasonal_relevance=float(row["seasonal_relevance"]),
            )

    # ── Serve ──────────────────────────────────────────────────────────────────

    def get_user_features(self, user_id: str) -> UserFeatures | None:
        """Retrieve pre-computed user feature vector. O(1)."""
        return self._users.get(user_id)

    def get_product_features(self, product_id: str) -> ProductFeatures | None:
        """Retrieve pre-computed product feature vector. O(1)."""
        return self._products.get(product_id)

    def get_all_product_ids(self) -> list[str]:
        return list(self._products.keys())

    def get_all_user_ids(self) -> list[str]:
        return list(self._users.keys())

    def user_to_vector(self, user_id: str) -> np.ndarray | None:
        """Convert user features to a numpy vector for model inference."""
        f = self.get_user_features(user_id)
        if f is None:
            return None
        return np.array([
            f.recency_days,
            f.purchase_freq_90d,
            f.avg_order_value,
            f.lifetime_spend,
            f.price_sensitivity,
            f.sessions_7d,
            f.cart_abandon_rate,
            f.account_age_days,
            f.segment_encoded,
            f.purchase_count_30d,
            f.avg_discount_accepted,
            *f.category_affinities.values(),
        ], dtype=np.float32)

    def product_to_vector(self, product_id: str) -> np.ndarray | None:
        """Convert product features to a numpy vector for model inference."""
        f = self.get_product_features(product_id)
        if f is None:
            return None
        return np.array([
            f.base_price,
            f.current_price,
            f.cost_price,
            float(f.current_stock),
            f.sell_through_rate,
            float(f.days_on_shelf),
            float(f.reorder_threshold),
            f.avg_rating,
            float(f.review_count),
            f.trend_score,
            f.margin_score,
            float(f.is_new_launch),
            float(f.style_cluster),
            f.seasonal_relevance,
        ], dtype=np.float32)

    @property
    def is_built(self) -> bool:
        return self._is_built
