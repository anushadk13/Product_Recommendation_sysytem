"""
ranking/diversity.py
─────────────────────
Diversity penalty for the product ranking grid.

Problem: Without a penalty, the ranking engine tends to show 10 nearly-identical
products (e.g., 10 pairs of aviator sunglasses). This destroys user experience.

Solution: Maximal Marginal Relevance (MMR) — at each position, apply a penalty
proportional to how similar the candidate is to already-selected products.

Similarity is based on:
  1. Same style_cluster  (the biggest driver)
  2. Same category
  3. Similar price tier

The penalty is subtracted from the ranking engine's final_score before selection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from data.feature_store import ProductFeatures


PRICE_TIERS = [(0, 50), (50, 150), (150, 350), (350, 700), (700, float("inf"))]


def _price_tier(price: float) -> int:
    for i, (lo, hi) in enumerate(PRICE_TIERS):
        if lo <= price < hi:
            return i
    return len(PRICE_TIERS) - 1


def product_similarity(a: ProductFeatures, b: ProductFeatures) -> float:
    """
    Heuristic similarity score ∈ [0, 1] between two products.
      1.0 = identical / same cluster + category + tier
      0.0 = completely different
    """
    # Style cluster match (most important)
    cluster_match = 1.0 if a.style_cluster == b.style_cluster else 0.0

    # Category match
    cat_match = 1.0 if a.category == b.category else 0.0

    # Price tier match
    tier_a = _price_tier(a.current_price)
    tier_b = _price_tier(b.current_price)
    tier_match = 1.0 - (abs(tier_a - tier_b) / len(PRICE_TIERS))

    # Weighted sum
    sim = 0.50 * cluster_match + 0.30 * cat_match + 0.20 * tier_match
    return round(sim, 4)


def compute_diversity_penalty(
    candidate: ProductFeatures,
    selected: list[ProductFeatures],
    lambda_diversity: float = 0.15,
) -> float:
    """
    MMR-style diversity penalty.

    penalty = λ × max_similarity(candidate, already_selected)

    If no products selected yet: penalty = 0.
    If candidate is very similar to an already-selected product: high penalty.
    """
    if not selected:
        return 0.0

    max_sim = max(product_similarity(candidate, s) for s in selected)
    return round(lambda_diversity * max_sim, 4)


def mmr_rerank(
    candidates: list[tuple[str, float]],   # (product_id, score)
    product_features: dict[str, ProductFeatures],
    top_k: int = 20,
    lambda_diversity: float = 0.15,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance re-ranking.

    Args:
        candidates   : Ranked list of (product_id, score) pairs
        product_features: Feature lookup dict
        top_k        : Number of products to select
        lambda_diversity: Strength of diversity penalty

    Returns:
        Re-ranked list of (product_id, adjusted_score) pairs
    """
    selected: list[ProductFeatures] = []
    selected_ids: list[str] = []
    remaining = list(candidates)

    while remaining and len(selected_ids) < top_k:
        best_pid = None
        best_score = -np.inf

        for pid, score in remaining:
            feat = product_features.get(pid)
            if feat is None:
                continue
            penalty = compute_diversity_penalty(feat, selected, lambda_diversity)
            adjusted = score - penalty
            if adjusted > best_score:
                best_score = adjusted
                best_pid = pid

        if best_pid is None:
            break

        feat = product_features[best_pid]
        selected.append(feat)
        selected_ids.append(best_pid)

        # Remove from remaining
        remaining = [(pid, s) for pid, s in remaining if pid != best_pid]

    # Return tuples with final adjusted scores
    result = []
    for pid in selected_ids:
        original_score = next((s for p, s in candidates if p == pid), 0.0)
        feat = product_features.get(pid)
        if feat:
            penalty = compute_diversity_penalty(feat, [product_features[p] for p in selected_ids if p != pid], lambda_diversity)
            adjusted = max(0, original_score - penalty)
        else:
            adjusted = original_score
        result.append((pid, round(adjusted, 6)))

    return result
