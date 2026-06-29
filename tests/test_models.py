"""
tests/test_models.py
─────────────────────
Unit tests for all 4 ML models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from data.generator import generate_all
from models.propensity import PropensityModel
from models.price_elasticity import PriceElasticityModel
from models.inventory_pressure import InventoryPressureModel
from models.demand_forecast import DemandForecastModel


@pytest.fixture(scope="module")
def data():
    return generate_all(n_users=100, n_products=50, n_events=2000, n_experiments=500, seed=1)


class TestPropensityModel:
    def test_train(self, data):
        model = PropensityModel()
        metrics = model.train(data["events"], data["users"], data["products"], verbose=False)
        assert "val_auc" in metrics
        assert 0.5 <= metrics["val_auc"] <= 1.0

    def test_predict_range(self, data):
        model = PropensityModel()
        model.train(data["events"], data["users"], data["products"], verbose=False)
        from data.feature_store import FeatureStore
        store = FeatureStore()
        store.build(data["users"], data["products"], data["events"])
        user_vec = store.user_to_vector(data["users"]["user_id"].iloc[0])
        prod_vec = store.product_to_vector(data["products"]["product_id"].iloc[0])
        score = model.predict(user_vec, prod_vec)
        assert 0.0 <= score <= 1.0

    def test_batch_predict_shape(self, data):
        model = PropensityModel()
        model.train(data["events"], data["users"], data["products"], verbose=False)
        from data.feature_store import FeatureStore
        store = FeatureStore()
        store.build(data["users"], data["products"], data["events"])
        user_vec = store.user_to_vector(data["users"]["user_id"].iloc[0])
        prod_vecs = [store.product_to_vector(pid) for pid in data["products"]["product_id"].iloc[:5]]
        scores = model.predict_batch(user_vec, prod_vecs)
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestPriceElasticityModel:
    def test_train(self, data):
        model = PriceElasticityModel()
        metrics = model.train(data["experiments"], data["products"], verbose=False)
        assert "mse_control" in metrics
        assert "mse_treated" in metrics

    def test_uplift_nonnegative(self, data):
        model = PriceElasticityModel()
        model.train(data["experiments"], data["products"], verbose=False)
        pid = data["products"]["product_id"].iloc[0]
        uplift = model.predict_uplift(pid, discount_pct=0.2)
        assert uplift >= 0.0

    def test_optimal_discount_keys(self, data):
        model = PriceElasticityModel()
        model.train(data["experiments"], data["products"], verbose=False)
        pid = data["products"]["product_id"].iloc[0]
        result = model.optimal_discount(pid)
        assert "optimal_discount_pct" in result
        assert "expected_uplift" in result
        assert 0.0 <= result["optimal_discount_pct"] <= 0.5


class TestInventoryPressureModel:
    def test_train(self, data):
        model = InventoryPressureModel()
        metrics = model.train(data["products"], verbose=False)
        assert "val_mse" in metrics
        assert "val_auc_threshold" in metrics

    def test_batch_scores_range(self, data):
        model = InventoryPressureModel()
        model.train(data["products"], verbose=False)
        scores = model.predict_batch(data["products"])
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_urgency_labels(self):
        m = InventoryPressureModel()
        assert m.urgency_label(0.90) == "Critical"
        assert m.urgency_label(0.70) == "High"
        assert m.urgency_label(0.50) == "Medium"
        assert m.urgency_label(0.10) == "Low"

    def test_sale_badge_threshold(self):
        m = InventoryPressureModel()
        assert m.should_show_sale_badge(0.70) is True
        assert m.should_show_sale_badge(0.30) is False


class TestDemandForecastModel:
    def test_train(self, data):
        model = DemandForecastModel()
        metrics = model.train(data["events"], data["products"], verbose=False)
        assert "mae" in metrics

    def test_spike_score_range(self, data):
        model = DemandForecastModel()
        model.train(data["events"], data["products"], verbose=False)
        pid = data["products"]["product_id"].iloc[0]
        score = model.demand_spike_score(pid, predicted_demand=10.0)
        assert 0.0 <= score <= 1.0
