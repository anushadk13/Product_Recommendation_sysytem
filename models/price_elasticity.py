"""
models/price_elasticity.py
───────────────────────────
Model 2: Price Elasticity
"If we drop this product X%, how much does conversion lift?"

Approach: Causal ML (simplified CausalML / EconML-style)
  - Treatment  : discount_pct (0.0 → 0.5)
  - Outcome    : conversion_rate
  - Nuisance   : product-level confounders (category, base price, brand)

We fit a T-Learner:
  - μ₀(x) = XGBoost on control group (discount < 5%)
  - μ₁(x) = XGBoost on treated group (discount ≥ 5%)
  - CATE  = μ₁(x) - μ₀(x)  → conditional average treatment effect

Then expose:
  - predict_uplift(product_id, discount_pct) → expected conversion lift
  - optimal_discount(product_id) → argmax uplift / margin trade-off
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import math


MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "elasticity_model.joblib"


class PriceElasticityModel:
    """
    T-Learner causal model for price elasticity estimation.
    """

    def __init__(self):
        # μ₀: outcome model for low-discount (control)
        self._mu0 = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        # μ₁: outcome model for high-discount (treated)
        self._mu1 = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        self.is_trained = False
        self._product_meta: dict[str, dict] = {}   # cached product-level features

    # ── Feature Engineering ──────────────────────────────────────────────────

    @staticmethod
    def _product_to_features(products_df: pd.DataFrame) -> pd.DataFrame:
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        df = products_df[["base_price", "avg_rating", "review_count",
                           "margin_score", "trend_score", "is_new_launch"]].copy()
        df["category_encoded"] = products_df["category"].map(cat_map).fillna(0)
        return df

    def _build_training_data(
        self,
        experiments_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> tuple:
        prod_feats = self._product_to_features(products_df).set_index(products_df["product_id"])

        X_all, y_all, treat_all = [], [], []
        for _, row in experiments_df.iterrows():
            pid = row["product_id"]
            if pid not in prod_feats.index:
                continue
            p = prod_feats.loc[pid].values.astype(np.float32)
            discount = float(row["discount_pct"])
            conv = float(row["conversion_rate"])
            X_all.append(np.concatenate([p, [discount]]))
            y_all.append(conv)
            treat_all.append(1 if discount >= 0.05 else 0)

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        t = np.array(treat_all, dtype=np.int32)
        return X, y, t

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        experiments_df: pd.DataFrame,
        products_df: pd.DataFrame,
        verbose: bool = True,
    ) -> dict:
        if verbose:
            print("[PriceElasticity] Building training data...")

        X, y, t = self._build_training_data(experiments_df, products_df)

        # Split into control (t=0) and treated (t=1)
        X0, y0 = X[t == 0], y[t == 0]
        X1, y1 = X[t == 1], y[t == 1]

        X0_tr, X0_val, y0_tr, y0_val = train_test_split(X0, y0, test_size=0.2, random_state=42)
        X1_tr, X1_val, y1_tr, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=43)

        if verbose:
            print(f"  → Control group: {len(X0_tr)} train | Treated: {len(X1_tr)} train")
            print("[PriceElasticity] Training T-Learner (μ₀, μ₁)...")

        self._mu0.fit(X0_tr, y0_tr)
        self._mu1.fit(X1_tr, y1_tr)
        self.is_trained = True

        mse0 = mean_squared_error(y0_val, self._mu0.predict(X0_val))
        mse1 = mean_squared_error(y1_val, self._mu1.predict(X1_val))

        # Cache product meta for inference
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        for _, row in products_df.iterrows():
            self._product_meta[row["product_id"]] = {
                "base_price": row["base_price"],
                "avg_rating": row["avg_rating"],
                "review_count": row["review_count"],
                "margin_score": row["margin_score"],
                "trend_score": row["trend_score"],
                "is_new_launch": row["is_new_launch"],
                "category_encoded": cat_map.get(row["category"], 0),
                "current_price": row["current_price"],
                "cost_price": row["cost_price"],
            }

        metrics = {"mse_control": round(mse0, 6), "mse_treated": round(mse1, 6)}
        if verbose:
            print(f"  → MSE control: {mse0:.6f} | MSE treated: {mse1:.6f}")
        return metrics

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_uplift(self, product_id: str, discount_pct: float) -> float:
        """
        Returns expected conversion lift from applying discount_pct to product.
        CATE = μ₁(x, discount) - μ₀(x, 0)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        meta = self._product_meta.get(product_id)
        if meta is None:
            return 0.0

        p = np.array([
            meta["base_price"], meta["avg_rating"], meta["review_count"],
            meta["margin_score"], meta["trend_score"], meta["is_new_launch"],
            meta["category_encoded"],
        ], dtype=np.float32)

        x_treated = np.concatenate([p, [discount_pct]]).reshape(1, -1)
        x_control = np.concatenate([p, [0.0]]).reshape(1, -1)

        cate = float(self._mu1.predict(x_treated)[0]) - float(self._mu0.predict(x_control)[0])
        return round(max(0.0, cate), 4)

    def optimal_discount(
        self,
        product_id: str,
        max_discount: float = 0.5,
        margin_weight: float = 0.5,
    ) -> dict:
        """
        Find the discount that maximizes the uplift / margin trade-off.
        Returns dict with optimal_discount_pct and expected_uplift.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        meta = self._product_meta.get(product_id)
        if meta is None:
            return {"optimal_discount_pct": 0.0, "expected_uplift": 0.0, "expected_margin_loss": 0.0}

        best_score = -np.inf
        best_discount = 0.0
        best_uplift = 0.0

        for d in np.arange(0.0, max_discount + 0.01, 0.05):
            uplift = self.predict_uplift(product_id, d)
            margin_loss = d * (meta["current_price"] - meta["cost_price"]) / (meta["current_price"] + 1e-6)
            score = (1 - margin_weight) * uplift - margin_weight * margin_loss
            if score > best_score:
                best_score = score
                best_discount = round(d, 2)
                best_uplift = uplift

        return {
            "optimal_discount_pct": best_discount,
            "expected_uplift": round(best_uplift, 4),
            "expected_margin_loss": round(best_discount * 0.5, 4),
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mu0": self._mu0, "mu1": self._mu1, "meta": self._product_meta}, path)
        print(f"[PriceElasticity] Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> None:
        data = joblib.load(path)
        self._mu0 = data["mu0"]
        self._mu1 = data["mu1"]
        self._product_meta = data["meta"]
        self.is_trained = True
        print(f"[PriceElasticity] Model loaded ← {path}")
