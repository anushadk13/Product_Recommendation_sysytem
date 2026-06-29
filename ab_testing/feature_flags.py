"""
ab_testing/feature_flags.py
────────────────────────────
A/B testing weight configurations for the ranking engine.

Different business presets tune the ranking formula weights for different goals:

  "balanced"  — General e-commerce (default)
  "lenskart"  — End-of-season: prioritize moving inventory
  "dyson"     — New launch: prioritize trending / demand spike
  "amazon"    — Maximize conversion: propensity-heavy

Each variant can be assigned to a user bucket (hash-based assignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WeightPreset = Literal["balanced", "lenskart", "dyson", "amazon", "clearance"]


@dataclass(frozen=True)
class WeightConfig:
    """
    Weights for the ranking formula.
    All weights should sum to ~1.0.
    """
    name: str
    description: str
    propensity: float      # W for purchase_propensity_score
    inventory: float       # W for inventory_pressure_score
    margin: float          # W for margin_score
    trend: float           # W for trend_score
    demand_spike: float    # W for demand_spike_score
    diversity: float       # λ for MMR diversity penalty (subtracted, not summed)

    def __post_init__(self):
        total = self.propensity + self.inventory + self.margin + self.trend + self.demand_spike
        assert 0.95 <= total <= 1.05, f"Weights must sum to ~1.0, got {total:.3f}"


# ─── Weight Presets ────────────────────────────────────────────────────────────

WEIGHT_CONFIGS: dict[str, WeightConfig] = {

    "balanced": WeightConfig(
        name="balanced",
        description="General e-commerce: balanced trade-off across all signals",
        propensity=0.35,
        inventory=0.25,
        margin=0.20,
        trend=0.10,
        demand_spike=0.10,
        diversity=0.15,
    ),

    "lenskart": WeightConfig(
        name="lenskart",
        description="End-of-season clearance: maximise inventory movement",
        propensity=0.25,
        inventory=0.45,     # ← inventory pressure dominates
        margin=0.15,
        trend=0.05,
        demand_spike=0.10,
        diversity=0.20,
    ),

    "dyson": WeightConfig(
        name="dyson",
        description="New product launch: ride the trend and demand spike",
        propensity=0.30,
        inventory=0.10,
        margin=0.15,
        trend=0.25,         # ← trend score dominates
        demand_spike=0.20,  # ← demand spike matters for pre-positioning
        diversity=0.10,
    ),

    "amazon": WeightConfig(
        name="amazon",
        description="High-volume marketplace: maximize conversion probability",
        propensity=0.50,    # ← propensity dominates
        inventory=0.15,
        margin=0.20,
        trend=0.10,
        demand_spike=0.05,
        diversity=0.10,
    ),

    "clearance": WeightConfig(
        name="clearance",
        description="Flash sale / clearance event: move critical inventory fast",
        propensity=0.20,
        inventory=0.50,     # ← maximum inventory pressure weight
        margin=0.10,
        trend=0.10,
        demand_spike=0.10,
        diversity=0.25,     # ← high diversity to show wide range of items
    ),
}


def get_weight_config(preset: str = "balanced") -> WeightConfig:
    """Retrieve weight config by preset name. Defaults to 'balanced'."""
    return WEIGHT_CONFIGS.get(preset, WEIGHT_CONFIGS["balanced"])


def assign_ab_variant(user_id: str, experiment_name: str = "ranking_weights") -> str:
    """
    Deterministic A/B variant assignment based on user_id hash.
    Same user always gets the same variant within an experiment.

    Splits users into:
      40% balanced | 20% lenskart | 20% dyson | 20% amazon
    """
    bucket = hash(f"{experiment_name}:{user_id}") % 100
    if bucket < 40:
        return "balanced"
    elif bucket < 60:
        return "lenskart"
    elif bucket < 80:
        return "dyson"
    else:
        return "amazon"


def list_presets() -> list[dict]:
    """List all available presets with their weight breakdown."""
    return [
        {
            "name": cfg.name,
            "description": cfg.description,
            "weights": {
                "propensity": cfg.propensity,
                "inventory": cfg.inventory,
                "margin": cfg.margin,
                "trend": cfg.trend,
                "demand_spike": cfg.demand_spike,
                "diversity_lambda": cfg.diversity,
            }
        }
        for cfg in WEIGHT_CONFIGS.values()
    ]
