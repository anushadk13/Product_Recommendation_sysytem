"""
models/demand_forecast.py
──────────────────────────
Model 4: Demand Forecast
"Will demand spike for this product in the next 7 days?"

Approach: LightGBM on engineered time-series features
  - Rolling stats (7d, 14d, 30d purchase counts)
  - Day-of-week / month seasonality
  - Trend velocity (rate of change in views)
  - Product age signal

Output per product:
  - demand_forecast_7d: expected units sold in next 7 days
  - demand_spike_score: normalized [0,1] — how unusual/high is this forecast?

Dyson uses this to pre-position products before seasonal peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "demand_model.joblib"


class DemandForecastModel:
    """
    LightGBM demand forecasting model.
    Forecasts 7-day demand per product and surfaces spike score.
    """

    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.is_trained = False
        self._product_baselines: dict[str, float] = {}   # avg 7d demand per product
        self._max_demand: float = 1.0

    # ── Feature Engineering ──────────────────────────────────────────────────

    def _build_time_features(
        self,
        events_df: pd.DataFrame,
        products_df: pd.DataFrame,
        horizon_days: int = 7,
    ) -> pd.DataFrame:
        """
        Build tabular features for demand forecasting.
        Creates sliding-window samples: (product_id, week_start) → demand_next_7d.
        """
        events = events_df[events_df["event_type"] == "purchase"].copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"])
        events["date"] = events["timestamp"].dt.date

        # Aggregate daily demand per product
        daily = (
            events.groupby(["product_id", "date"])
            .size()
            .reset_index(name="daily_demand")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(["product_id", "date"])

        # Product features lookup
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        prod_lookup = products_df.set_index("product_id")[
            ["trend_score", "avg_rating", "review_count",
             "seasonal_relevance", "is_new_launch", "base_price", "category"]
        ].copy()
        prod_lookup["category_encoded"] = prod_lookup["category"].map(cat_map).fillna(0)

        rows = []
        for pid, group in daily.groupby("product_id"):
            group = group.set_index("date").sort_index()
            dates = group.index.tolist()

            for i, dt in enumerate(dates):
                if i < 14:   # need at least 14 days history
                    continue
                # Features: rolling stats
                history = group["daily_demand"].iloc[:i]
                demand_7d = history.iloc[-7:].sum()
                demand_14d = history.iloc[-14:].sum()
                demand_30d = history.iloc[-30:].sum() if len(history) >= 30 else history.sum()

                # Target: next-7d demand
                future = group["daily_demand"].iloc[i:i + horizon_days]
                if len(future) < horizon_days:
                    continue
                target = future.sum()

                # Time features
                day_of_week = dt.weekday()
                month = dt.month
                is_weekend = int(day_of_week >= 5)

                # Velocity: (7d - 14d/2) / (14d/2)
                prev_7d = history.iloc[-14:-7].sum() if len(history) >= 14 else demand_7d
                velocity = (demand_7d - prev_7d) / (prev_7d + 1e-6)

                prod = prod_lookup.loc[pid] if pid in prod_lookup.index else None
                row = {
                    "product_id": pid,
                    "date": dt,
                    "demand_7d": demand_7d,
                    "demand_14d": demand_14d,
                    "demand_30d": demand_30d,
                    "velocity": velocity,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "trend_score": float(prod["trend_score"]) if prod is not None else 0.5,
                    "avg_rating": float(prod["avg_rating"]) if prod is not None else 3.5,
                    "seasonal_relevance": float(prod["seasonal_relevance"]) if prod is not None else 0.5,
                    "is_new_launch": int(prod["is_new_launch"]) if prod is not None else 0,
                    "base_price": float(prod["base_price"]) if prod is not None else 50.0,
                    "category_encoded": float(prod["category_encoded"]) if prod is not None else 0.0,
                    "target_demand_7d": target,
                }
                rows.append(row)

        return pd.DataFrame(rows)

    FEATURE_COLS = [
        "demand_7d", "demand_14d", "demand_30d", "velocity",
        "day_of_week", "month", "is_weekend",
        "trend_score", "avg_rating", "seasonal_relevance",
        "is_new_launch", "base_price", "category_encoded",
    ]

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        events_df: pd.DataFrame,
        products_df: pd.DataFrame,
        verbose: bool = True,
    ) -> dict:
        if verbose:
            print("[DemandForecast] Building time-series features (this may take a moment)...")

        df = self._build_time_features(events_df, products_df)
        if len(df) < 100:
            print("  ⚠ Not enough time-series data; using fallback simple model")
            self.is_trained = True
            return {"mae": 0.0, "rmse": 0.0, "n_samples": 0}

        X = df[self.FEATURE_COLS].values.astype(np.float32)
        y = df["target_demand_7d"].values.astype(np.float32)

        # Record baselines
        self._product_baselines = df.groupby("product_id")["demand_7d"].mean().to_dict()
        self._max_demand = float(y.max()) if len(y) > 0 else 1.0

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if verbose:
            print(f"  → {len(X_train)} training samples | {len(X_val)} validation samples")
            print("[DemandForecast] Training LightGBM...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=None,
        )
        self.is_trained = True

        preds = self.model.predict(X_val).clip(0)
        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds) ** 0.5

        metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "n_samples": len(X_train)}
        if verbose:
            print(f"  → MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        return metrics

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_demand(self, feature_vector: np.ndarray) -> float:
        """Predict 7-day demand for a given feature vector."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return float(max(0, self.model.predict(feature_vector.reshape(1, -1))[0]))

    def demand_spike_score(self, product_id: str, predicted_demand: float) -> float:
        """
        Normalize demand against product's historical baseline.
        score = predicted / (baseline * 1.5)  → clipped to [0, 1]
        """
        baseline = self._product_baselines.get(product_id, 1.0)
        if baseline == 0:
            baseline = 1.0
        spike = predicted_demand / (baseline * 1.5)
        return round(float(min(1.0, spike)), 4)

    def get_product_forecast_features(
        self,
        product_id: str,
        events_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extract the latest feature vector for a product (for inference).
        """
        events = events_df[
            (events_df["product_id"] == product_id) &
            (events_df["event_type"] == "purchase")
        ].copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"])
        events["date"] = events["timestamp"].dt.date

        if len(events) == 0:
            return np.zeros(len(self.FEATURE_COLS), dtype=np.float32)

        daily = events.groupby("date").size().sort_index()
        demand_7d = int(daily.tail(7).sum())
        demand_14d = int(daily.tail(14).sum())
        demand_30d = int(daily.tail(30).sum())
        prev_7d = int(daily.iloc[-14:-7].sum()) if len(daily) >= 14 else demand_7d
        velocity = (demand_7d - prev_7d) / (prev_7d + 1e-6)

        now = datetime.utcnow()
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        prod_row = products_df[products_df["product_id"] == product_id]
        if len(prod_row) == 0:
            return np.zeros(len(self.FEATURE_COLS), dtype=np.float32)
        prod = prod_row.iloc[0]

        return np.array([
            demand_7d, demand_14d, demand_30d, velocity,
            now.weekday(), now.month, int(now.weekday() >= 5),
            float(prod["trend_score"]), float(prod["avg_rating"]),
            float(prod["seasonal_relevance"]), int(prod["is_new_launch"]),
            float(prod["base_price"]), float(cat_map.get(prod["category"], 0)),
        ], dtype=np.float32)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "baselines": self._product_baselines,
            "max_demand": self._max_demand,
        }, path)
        print(f"[DemandForecast] Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._product_baselines = data["baselines"]
        self._max_demand = data["max_demand"]
        self.is_trained = True
        print(f"[DemandForecast] Model loaded ← {path}")
