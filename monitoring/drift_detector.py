"""
monitoring/drift_detector.py
─────────────────────────────
Feature drift detection — Evidently AI-style (but self-contained).

Detects when the distribution of features at inference time has shifted
significantly from the training distribution. This is critical because:
  - User behaviour changes seasonally
  - Product catalogue evolves
  - External shocks (viral moment, competitor sale) spike patterns

Implements:
  1. Population Stability Index (PSI) — for continuous features
  2. Chi-square test — for categorical features
  3. Z-score alert — for scalar metrics (avg propensity score, etc.)

Usage:
    detector = DriftDetector()
    detector.fit_reference(training_features_df)
    alerts = detector.check(current_features_df)
    for alert in alerts:
        print(alert)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from scipy import stats


# ─── Thresholds ────────────────────────────────────────────────────────────────
PSI_WARNING  = 0.10   # slight drift
PSI_CRITICAL = 0.25   # significant drift → retrain model

DriftLevel = Literal["none", "warning", "critical"]


@dataclass
class DriftAlert:
    feature: str
    metric: str            # "PSI", "chi2", "zscore"
    value: float
    threshold: float
    level: DriftLevel
    message: str

    def __str__(self) -> str:
        emoji = {"none": "✅", "warning": "⚠️", "critical": "🚨"}[self.level]
        return f"{emoji} [{self.level.upper()}] {self.feature}: {self.metric}={self.value:.4f} (threshold={self.threshold})"


class DriftDetector:
    """
    Lightweight feature drift detector.
    Fits on training data distributions, then checks new batches.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._reference_distributions: dict[str, dict] = {}
        self._numeric_features: list[str] = []
        self._categorical_features: list[str] = []
        self.is_fitted = False

    # ── Fit Reference ─────────────────────────────────────────────────────────

    def fit_reference(self, df: pd.DataFrame) -> None:
        """
        Fit reference distributions from training data.
        Call this after training, before deploying to production.
        """
        self._numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in self._numeric_features:
            series = df[col].dropna()
            bins = np.percentile(series, np.linspace(0, 100, self.n_bins + 1))
            bins = np.unique(bins)
            hist, bin_edges = np.histogram(series, bins=bins)
            self._reference_distributions[col] = {
                "type": "numeric",
                "bins": bin_edges,
                "hist": hist / hist.sum(),   # normalize to proportions
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

        for col in self._categorical_features:
            value_counts = df[col].value_counts(normalize=True).to_dict()
            self._reference_distributions[col] = {
                "type": "categorical",
                "proportions": value_counts,
            }

        self.is_fitted = True
        print(f"[DriftDetector] Reference fitted on {len(df)} rows | "
              f"{len(self._numeric_features)} numeric + {len(self._categorical_features)} categorical features")

    # ── Check ─────────────────────────────────────────────────────────────────

    def check(self, current_df: pd.DataFrame, verbose: bool = True) -> list[DriftAlert]:
        """
        Compare current data distribution against reference.
        Returns list of DriftAlerts (empty if no drift).
        """
        if not self.is_fitted:
            raise RuntimeError("DriftDetector not fitted. Call fit_reference() first.")

        alerts: list[DriftAlert] = []

        # Numeric: PSI
        for col in self._numeric_features:
            if col not in current_df.columns:
                continue
            ref = self._reference_distributions[col]
            if ref["type"] != "numeric":
                continue
            current = current_df[col].dropna()
            if len(current) == 0:
                continue

            psi = self._compute_psi(current, ref["bins"], ref["hist"])
            level = self._psi_level(psi)
            if level != "none":
                alerts.append(DriftAlert(
                    feature=col, metric="PSI", value=psi,
                    threshold=PSI_WARNING, level=level,
                    message=f"Distribution shift detected in '{col}' (PSI={psi:.4f})",
                ))

        # Categorical: Chi-squared
        for col in self._categorical_features:
            if col not in current_df.columns:
                continue
            ref = self._reference_distributions[col]
            alert = self._check_categorical(col, current_df[col], ref["proportions"])
            if alert:
                alerts.append(alert)

        if verbose:
            if alerts:
                print(f"\n[DriftDetector] {len(alerts)} drift alert(s) detected:")
                for a in alerts:
                    print(f"  {a}")
            else:
                print("[DriftDetector] ✅ No drift detected")

        return alerts

    # ── PSI ───────────────────────────────────────────────────────────────────

    def _compute_psi(
        self,
        current: pd.Series,
        bins: np.ndarray,
        ref_proportions: np.ndarray,
    ) -> float:
        """Population Stability Index."""
        curr_hist, _ = np.histogram(current, bins=bins)
        curr_proportions = curr_hist / (curr_hist.sum() + 1e-10)

        # Clip to avoid log(0)
        p = np.clip(ref_proportions, 1e-10, None)
        q = np.clip(curr_proportions, 1e-10, None)

        psi = np.sum((q - p) * np.log(q / p))
        return float(psi)

    @staticmethod
    def _psi_level(psi: float) -> DriftLevel:
        if psi >= PSI_CRITICAL:
            return "critical"
        if psi >= PSI_WARNING:
            return "warning"
        return "none"

    # ── Chi-Square ────────────────────────────────────────────────────────────

    def _check_categorical(
        self,
        col: str,
        current_series: pd.Series,
        ref_proportions: dict,
    ) -> DriftAlert | None:
        current_counts = current_series.value_counts()
        n = len(current_series)
        categories = list(ref_proportions.keys())

        expected = np.array([ref_proportions.get(c, 1e-10) * n for c in categories])
        observed = np.array([current_counts.get(c, 0) for c in categories])

        if expected.sum() == 0:
            return None

        chi2_stat, p_value = stats.chisquare(observed + 1e-10, expected + 1e-10)

        if p_value < 0.01:
            return DriftAlert(
                feature=col, metric="chi2_pvalue", value=round(p_value, 6),
                threshold=0.01, level="critical",
                message=f"Categorical distribution shift in '{col}' (p={p_value:.6f})",
            )
        if p_value < 0.05:
            return DriftAlert(
                feature=col, metric="chi2_pvalue", value=round(p_value, 6),
                threshold=0.05, level="warning",
                message=f"Possible categorical drift in '{col}' (p={p_value:.6f})",
            )
        return None

    # ── Score Monitoring ──────────────────────────────────────────────────────

    def check_score_distribution(
        self,
        current_scores: np.ndarray,
        reference_mean: float,
        reference_std: float,
        feature_name: str = "recommendation_score",
    ) -> DriftAlert | None:
        """
        Z-score alert on recommendation score distribution.
        Useful for monitoring output drift (when model degrades).
        """
        current_mean = float(np.mean(current_scores))
        z = abs(current_mean - reference_mean) / (reference_std + 1e-10)

        if z > 3.0:
            return DriftAlert(
                feature=feature_name, metric="zscore", value=round(z, 4),
                threshold=3.0, level="critical",
                message=f"Score distribution anomaly: mean shifted {z:.1f}σ from baseline",
            )
        if z > 2.0:
            return DriftAlert(
                feature=feature_name, metric="zscore", value=round(z, 4),
                threshold=2.0, level="warning",
                message=f"Score distribution warning: mean shifted {z:.1f}σ from baseline",
            )
        return None

    def summary(self) -> dict:
        """Return a summary of the reference distribution statistics."""
        return {
            "n_numeric_features": len(self._numeric_features),
            "n_categorical_features": len(self._categorical_features),
            "features_monitored": self._numeric_features + self._categorical_features,
            "psi_thresholds": {"warning": PSI_WARNING, "critical": PSI_CRITICAL},
        }
