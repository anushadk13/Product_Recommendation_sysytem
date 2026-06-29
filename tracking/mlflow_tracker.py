"""
tracking/mlflow_tracker.py
───────────────────────────
MLflow experiment tracking wrapper.

Tracks:
  - Model training runs (params, metrics, artifacts)
  - Ranking engine configuration (weight presets)
  - A/B experiment results

Usage:
    tracker = MLflowTracker(experiment_name="merchandising_v1")
    with tracker.start_run("propensity_model") as run:
        tracker.log_params({"n_estimators": 300, "max_depth": 6})
        tracker.log_metrics({"val_auc": 0.82})
        tracker.log_model_summary("XGBoost propensity model")
"""

from __future__ import annotations

import os
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Lightweight MLflow wrapper (no server required) ──────────────────────────
# Falls back to local file logging if MLflow server is unavailable.

class MLflowTracker:
    """
    MLflow-compatible experiment tracker.
    Uses mlflow if available, otherwise writes structured logs locally.
    """

    def __init__(
        self,
        experiment_name: str = "personalized_merchandising",
        tracking_uri: str = "sqlite:///mlruns.db",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._use_mlflow = False
        self._run_id: str | None = None
        self._current_run_name: str = ""
        self._log_dir = Path(tracking_uri) / experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._runs: list[dict] = []

        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._use_mlflow = True
            print(f"[MLflow] ✓ Connected to tracking server at {tracking_uri}")
        except ImportError:
            self._mlflow = None
            print(f"[MLflow] ⚠ mlflow not installed — using local file logging at {self._log_dir}")

    @contextmanager
    def start_run(self, run_name: str):
        """Context manager for a training run."""
        self._current_run_name = run_name
        run_data = {
            "run_name": run_name,
            "experiment": self.experiment_name,
            "start_time": datetime.utcnow().isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
        }
        self._current_run_data = run_data

        if self._use_mlflow:
            with self._mlflow.start_run(run_name=run_name) as mlflow_run:
                self._run_id = mlflow_run.info.run_id
                yield self
        else:
            yield self

        run_data["end_time"] = datetime.utcnow().isoformat()
        self._runs.append(run_data)
        self._flush_run(run_data)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self._current_run_data["params"].update(params)
        if self._use_mlflow:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics."""
        self._current_run_data["metrics"].update(metrics)
        if self._use_mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def log_model_summary(self, summary: str) -> None:
        """Log a text summary of the model."""
        self._current_run_data["tags"]["model_summary"] = summary
        if self._use_mlflow:
            self._mlflow.set_tag("model_summary", summary[:250])

    def log_weight_config(self, preset_name: str, weights: dict) -> None:
        """Log the ranking engine weight configuration."""
        self.log_params({"weight_preset": preset_name})
        self.log_params({f"w_{k}": v for k, v in weights.items()})

    def _flush_run(self, run_data: dict) -> None:
        """Write run data to local JSON file."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = self._log_dir / f"{run_data['run_name']}_{ts}.json"
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
                
        with open(fname, "w") as f:
            json.dump(run_data, f, indent=2, cls=NpEncoder)

    def print_summary(self) -> None:
        """Print all logged runs to console."""
        print(f"\n{'='*60}")
        print(f"MLflow Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        for run in self._runs:
            print(f"\n  Run: {run['run_name']}")
            print(f"    Params  : {run['params']}")
            print(f"    Metrics : {run['metrics']}")
        print(f"\n  Logs written to: {self._log_dir}")
        print(f"{'='*60}\n")

    def get_best_run(self, metric: str, mode: str = "max") -> dict | None:
        """Return the run with the best value for a given metric."""
        eligible = [r for r in self._runs if metric in r["metrics"]]
        if not eligible:
            return None
        key = lambda r: r["metrics"][metric]
        return max(eligible, key=key) if mode == "max" else min(eligible, key=key)
