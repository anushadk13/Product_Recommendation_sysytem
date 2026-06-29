"""
tests/test_ranking.py
──────────────────────
Unit tests for the ranking engine, diversity, and A/B testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.generator import generate_all
from data.feature_store import FeatureStore
from ab_testing.feature_flags import get_weight_config, assign_ab_variant, list_presets, WEIGHT_CONFIGS
from ranking.diversity import product_similarity, compute_diversity_penalty, mmr_rerank


@pytest.fixture(scope="module")
def full_system():
    data = generate_all(n_users=100, n_products=50, n_events=2000, n_experiments=500, seed=2)
    store = FeatureStore()
    store.build(data["users"], data["products"], data["events"])

    from models.propensity import PropensityModel
    from models.price_elasticity import PriceElasticityModel
    from models.inventory_pressure import InventoryPressureModel
    from models.demand_forecast import DemandForecastModel
    from ranking.engine import RankingEngine

    prop = PropensityModel()
    prop.train(data["events"], data["users"], data["products"], verbose=False)
    elast = PriceElasticityModel()
    elast.train(data["experiments"], data["products"], verbose=False)
    inv = InventoryPressureModel()
    inv.train(data["products"], verbose=False)
    dem = DemandForecastModel()
    dem.train(data["events"], data["products"], verbose=False)

    engine = RankingEngine(store, prop, elast, inv, dem, data["products"], data["events"])
    return {"engine": engine, "store": store, "data": data}


class TestWeightConfigs:
    def test_all_presets_exist(self):
        for preset in ["balanced", "lenskart", "dyson", "amazon", "clearance"]:
            cfg = get_weight_config(preset)
            assert cfg.name == preset

    def test_weights_sum_to_one(self):
        for name, cfg in WEIGHT_CONFIGS.items():
            total = cfg.propensity + cfg.inventory + cfg.margin + cfg.trend + cfg.demand_spike
            assert 0.95 <= total <= 1.05, f"Weights for '{name}' sum to {total}"

    def test_list_presets_returns_all(self):
        presets = list_presets()
        assert len(presets) == len(WEIGHT_CONFIGS)

    def test_ab_assignment_deterministic(self):
        v1 = assign_ab_variant("U00001")
        v2 = assign_ab_variant("U00001")
        assert v1 == v2

    def test_ab_assignment_distributes(self):
        variants = [assign_ab_variant(f"U{i:05d}") for i in range(200)]
        unique = set(variants)
        assert len(unique) > 1   # at least 2 variants assigned


class TestDiversity:
    def test_identical_products_max_similarity(self, full_system):
        store = full_system["store"]
        pid = full_system["data"]["products"]["product_id"].iloc[0]
        pf = store.get_product_features(pid)
        sim = product_similarity(pf, pf)
        assert sim >= 0.8

    def test_diversity_penalty_grows_with_lambda(self, full_system):
        store = full_system["store"]
        pids = full_system["data"]["products"]["product_id"].iloc[:5].tolist()
        pfs = [store.get_product_features(p) for p in pids if store.get_product_features(p)]
        if len(pfs) < 2:
            pytest.skip("Not enough products")
        candidate = pfs[0]
        selected = pfs[1:]
        p1 = compute_diversity_penalty(candidate, selected, lambda_diversity=0.1)
        p2 = compute_diversity_penalty(candidate, selected, lambda_diversity=0.3)
        assert p2 >= p1

    def test_mmr_rerank_returns_top_k(self, full_system):
        store = full_system["store"]
        pids = full_system["data"]["products"]["product_id"].iloc[:20].tolist()
        candidates = [(pid, float(i) / 20) for i, pid in enumerate(reversed(pids))]
        pf_lookup = {pid: store.get_product_features(pid) for pid in pids}
        pf_lookup = {k: v for k, v in pf_lookup.items() if v is not None}
        result = mmr_rerank(candidates, pf_lookup, top_k=10)
        assert len(result) == min(10, len(pf_lookup))


class TestRankingEngine:
    def test_recommend_returns_results(self, full_system):
        engine = full_system["engine"]
        uid = full_system["data"]["users"]["user_id"].iloc[0]
        result = engine.recommend(uid, top_k=10)
        assert len(result.products) > 0
        assert result.latency_ms > 0

    def test_recommend_ranks_sequential(self, full_system):
        engine = full_system["engine"]
        uid = full_system["data"]["users"]["user_id"].iloc[0]
        result = engine.recommend(uid, top_k=5)
        ranks = [p.rank for p in result.products]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_score_pair_valid(self, full_system):
        engine = full_system["engine"]
        uid = full_system["data"]["users"]["user_id"].iloc[0]
        pid = full_system["data"]["products"]["product_id"].iloc[0]
        result = engine.score_pair(uid, pid)
        assert "final_score" in result
        assert 0.0 <= result["final_score"] <= 1.5

    def test_recommend_different_presets(self, full_system):
        engine = full_system["engine"]
        uid = full_system["data"]["users"]["user_id"].iloc[0]
        r1 = engine.recommend(uid, top_k=5, weight_preset="balanced")
        r2 = engine.recommend(uid, top_k=5, weight_preset="lenskart")
        # Different presets may produce different orderings
        assert r1.weights_used == "balanced"
        assert r2.weights_used == "lenskart"
