"""
models/propensity.py
─────────────────────
Model 1: Purchase Propensity
"Will THIS user buy THIS product in the next 24 hours?"

Architecture: Two-Tower approach
  - User Tower  : XGBoost on user behavioral features → user embedding score
  - Product Tower: XGBoost on product features → product score
  - Final score: dot-product fusion of both towers

In production: Two actual embedding towers (neural nets via TF/PyTorch),
but here we use XGBoost feature interactions which capture the same signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "propensity_model.joblib"
SCALER_PATH = MODEL_DIR / "propensity_scaler.joblib"


class PropensityModel:
    """
    Purchase propensity model.
    Predicts P(user buys product | features) using XGBoost.
    """

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names: list[str] = []

    # ── Feature Engineering ──────────────────────────────────────────────────

    @staticmethod
    def _build_user_features(users_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize user features for training."""
        cat_affinities = [c for c in users_df.columns if c.startswith("affinity_")]
        base_cols = [
            "recency_days", "purchase_freq_90d", "avg_order_value",
            "lifetime_spend", "price_sensitivity", "sessions_7d",
            "cart_abandon_rate", "account_age_days",
        ]
        segment_map = {"New": 0, "Lapsed": 1, "Regular": 2, "VIP": 3}
        df = users_df[base_cols + cat_affinities].copy()
        df["segment_encoded"] = users_df["segment"].map(segment_map).fillna(0)
        return df

    @staticmethod
    def _build_product_features(products_df: pd.DataFrame) -> pd.DataFrame:
        """Extract product features."""
        cols = [
            "base_price", "current_price", "sell_through_rate",
            "days_on_shelf", "avg_rating", "review_count",
            "trend_score", "margin_score", "is_new_launch", "seasonal_relevance",
        ]
        cat_map = {"Eyewear": 0, "Electronics": 1, "Home": 2, "Fashion": 3, "Beauty": 4, "Sports": 5}
        df = products_df[cols].copy()
        df["category_encoded"] = products_df["category"].map(cat_map).fillna(0)
        return df

    def _build_training_features(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct (user_features ⊕ product_features, label) pairs.
        Label = 1 if event_type == 'purchase', else 0.
        """
        user_feats = self._build_user_features(users_df).set_index(users_df["user_id"])
        prod_feats = self._build_product_features(products_df).set_index(products_df["product_id"])

        # Sample negative examples (views / wishlists that did not convert)
        positives = events_df[events_df["event_type"] == "purchase"].copy()
        negatives = events_df[events_df["event_type"].isin(["view", "wishlist"])].copy()

        # Balance dataset
        n_pos = len(positives)
        n_neg = min(n_pos * 3, len(negatives))
        negatives = negatives.sample(n=n_neg, random_state=42)

        samples = pd.concat([positives, negatives], ignore_index=True)
        samples["label"] = (samples["event_type"] == "purchase").astype(int)

        rows = []
        labels = []
        for _, row in samples.iterrows():
            uid, pid = row["user_id"], row["product_id"]
            if uid not in user_feats.index or pid not in prod_feats.index:
                continue
            u = user_feats.loc[uid].values.astype(np.float32)
            p = prod_feats.loc[pid].values.astype(np.float32)

            # Cross features (user × product interactions)
            price_ratio = u[2] / (p[1] + 1e-6)  # avg_order_value / current_price
            cross = np.array([price_ratio], dtype=np.float32)

            rows.append(np.concatenate([u, p, cross]))
            labels.append(row["label"])

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Store feature count for inference
        self.feature_names = (
            [f"u_{i}" for i in range(user_feats.shape[1])] +
            [f"p_{i}" for i in range(prod_feats.shape[1])] +
            ["price_ratio"]
        )
        return X, y

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
        verbose: bool = True,
    ) -> dict:
        """Train the propensity model. Returns training metrics."""
        if verbose:
            print("[Propensity] Building training features...")

        X, y = self._build_training_features(events_df, users_df, products_df)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        if verbose:
            print(f"  → Train: {X_train.shape}, Val: {X_val.shape} | Positives: {y.sum()}/{len(y)}")
            print("[Propensity] Training XGBoost...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.is_trained = True

        val_preds = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)

        metrics = {"val_auc": round(auc, 4), "n_train": len(X_train), "n_val": len(X_val)}
        if verbose:
            print(f"  → Validation AUC: {auc:.4f}")

        return metrics

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, user_vec: np.ndarray, product_vec: np.ndarray) -> float:
        """
        Score a single (user, product) pair.
        Returns P(purchase) in [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        u = user_vec[:11]   # first 11 base user features (approx)
        p = product_vec[:11]  # first 11 base product features (approx)

        # Rebuild cross features the same way as training
        price_ratio = u[2] / (p[1] + 1e-6)
        combined = np.concatenate([user_vec, product_vec, [price_ratio]], dtype=np.float32)

        # Pad/trim to match training shape
        target_len = self.model.n_features_in_
        if len(combined) < target_len:
            combined = np.pad(combined, (0, target_len - len(combined)))
        elif len(combined) > target_len:
            combined = combined[:target_len]

        combined_scaled = self.scaler.transform(combined.reshape(1, -1))
        return float(self.model.predict_proba(combined_scaled)[0, 1])

    def predict_batch(
        self,
        user_vec: np.ndarray,
        product_vecs: list[np.ndarray],
    ) -> np.ndarray:
        """
        Score one user against many products at once (efficient batch inference).
        Returns array of propensity scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        rows = []
        for p_vec in product_vecs:
            price_ratio = user_vec[2] / (p_vec[1] + 1e-6)
            combined = np.concatenate([user_vec, p_vec, [price_ratio]], dtype=np.float32)
            target_len = self.model.n_features_in_
            if len(combined) < target_len:
                combined = np.pad(combined, (0, target_len - len(combined)))
            elif len(combined) > target_len:
                combined = combined[:target_len]
            rows.append(combined)

        X = np.array(rows, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_names}, path)
        print(f"[Propensity] Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["features"]
        self.is_trained = True
        print(f"[Propensity] Model loaded ← {path}")
