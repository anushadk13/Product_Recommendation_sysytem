"""
models/inventory_pressure.py
──────────────────────────────
Model 3: Inventory Pressure Score
"Is this SKU at risk of becoming deadstock?"

Features:
  - current_stock / initial_stock      → stockout risk
  - sell_through_rate                  → velocity
  - days_on_shelf                      → ageing
  - reorder_threshold                  → buffer level
  - trend_score                        → future demand signal

Output: pressure_score ∈ [0, 1]
  - 0 = no urgency (selling well)
  - 1 = high urgency (deadstock risk → trigger sale badge)

Additionally surfaces:
  - should_discount: bool  (pressure > threshold)
  - urgency_label  : str   ("Low" | "Medium" | "High" | "Critical")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "inventory_model.joblib"

PRESSURE_THRESHOLD = 0.65   # above this → show sale badge
CRITICAL_THRESHOLD = 0.85   # above this → aggressive markdown


class InventoryPressureModel:
    """
    XGBoost model for deadstock / inventory pressure scoring.
    """

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    # ── Feature Engineering ──────────────────────────────────────────────────

    @staticmethod
    def _build_features(products_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame()

        # Core inventory signals
        df["stock_ratio"] = products_df["current_stock"] / (products_df["initial_stock"].clip(lower=1))
        df["sell_through_rate"] = products_df["sell_through_rate"]
        df["days_on_shelf"] = products_df["days_on_shelf"]
        df["days_normalized"] = (products_df["days_on_shelf"] / 180.0).clip(0, 1)

        # Threshold buffer: how close to reorder point?
        df["stock_buffer"] = (
            products_df["current_stock"] - products_df["reorder_threshold"]
        ).clip(lower=0) / (products_df["initial_stock"].clip(lower=1))

        # Demand signal
        df["trend_score"] = products_df["trend_score"]
        df["seasonal_relevance"] = products_df["seasonal_relevance"]

        # Price / margin
        df["margin_score"] = products_df["margin_score"]

        # Category encoded
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        df["category_encoded"] = products_df["category"].map(cat_map).fillna(0).values

        # Composite ageing score (high days + low throughput = deadstock signal)
        df["ageing_score"] = df["days_normalized"] * (1 - df["sell_through_rate"])

        return df.fillna(0).astype(np.float32)

    @staticmethod
    def _generate_labels(features_df: pd.DataFrame) -> np.ndarray:
        """
        Synthetic pressure labels: a deterministic formula that captures
        real-world deadstock logic. The model will learn this mapping.
        """
        pressure = (
            0.35 * features_df["ageing_score"] +
            0.30 * (1 - features_df["sell_through_rate"]) +
            0.20 * features_df["days_normalized"] +
            0.10 * (1 - features_df["trend_score"]) +
            0.05 * (1 - features_df["stock_buffer"].clip(0, 1))
        ).clip(0, 1)
        # Add noise to simulate real-world variance
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.03, size=len(pressure))
        return (pressure.values + noise).clip(0, 1).astype(np.float32)

    # ── Train ────────────────────────────────────────────────────────────────

    def train(self, products_df: pd.DataFrame, verbose: bool = True) -> dict:
        if verbose:
            print("[InventoryPressure] Building training features...")

        X = self._build_features(products_df)
        y = self._generate_labels(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y, test_size=0.2, random_state=42
        )

        if verbose:
            print(f"  → Train: {X_train.shape} | High-pressure products: {(y > PRESSURE_THRESHOLD).sum()}")
            print("[InventoryPressure] Training XGBoost regressor...")

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.is_trained = True

        val_preds = self.model.predict(X_val).clip(0, 1)
        mse = mean_squared_error(y_val, val_preds)

        # AUC on binarized threshold
        auc = roc_auc_score((y_val > PRESSURE_THRESHOLD).astype(int), val_preds)

        metrics = {"val_mse": round(mse, 6), "val_auc_threshold": round(auc, 4)}
        if verbose:
            print(f"  → Val MSE: {mse:.6f} | AUC (pressure badge): {auc:.4f}")
        return metrics

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_pressure(self, product_vec: np.ndarray) -> float:
        """
        Score a single product's inventory pressure.
        product_vec must be the output of _build_features for that product.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return float(self.model.predict(product_vec.reshape(1, -1)).clip(0, 1)[0])

    def predict_batch(self, products_df: pd.DataFrame) -> np.ndarray:
        """Score all products in a DataFrame. Returns array of pressure scores."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X = self._build_features(products_df)
        return self.model.predict(X.values).clip(0, 1)

    @staticmethod
    def urgency_label(score: float) -> str:
        if score >= CRITICAL_THRESHOLD:
            return "Critical"
        if score >= PRESSURE_THRESHOLD:
            return "High"
        if score >= 0.35:
            return "Medium"
        return "Low"

    @staticmethod
    def should_show_sale_badge(score: float) -> bool:
        return score >= PRESSURE_THRESHOLD

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model}, path)
        print(f"[InventoryPressure] Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.is_trained = True
        print(f"[InventoryPressure] Model loaded ← {path}")
