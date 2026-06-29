"""
file.py
────────
Main entry point for the Personalized Product Recommendation
+ Dynamic Merchandising System.

What this does when you run `python file.py`:
  1. Generate synthetic data (1000 users, 500 products, 50k events)
  2. Build in-memory feature store
  3. Train all 4 ML models (with MLflow tracking)
  4. Run the ranking engine for 3 sample users
  5. Demo price elasticity & sale badge logic
  6. Run drift detection
  7. Print a rich console summary
  8. Optionally start the FastAPI server

Run modes:
  python file.py            → pipeline demo (no server)
  python file.py --serve    → pipeline demo + start API server on :8080
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

# ─── Add project root to path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# ─── Colour helpers for console output ───────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    DIM    = "\033[2m"


def header(text: str) -> None:
    width = 70
    print(f"\n{C.CYAN}{'═' * width}")
    print(f"  {C.BOLD}{text}{C.RESET}{C.CYAN}")
    print(f"{'═' * width}{C.RESET}")


def step(text: str) -> None:
    print(f"\n{C.BLUE}▶{C.RESET} {text}")


def ok(text: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET} {text}")


def warn(text: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {text}")


def info(text: str) -> None:
    print(f"  {C.DIM}{text}{C.RESET}")


# ─── System builder (used by both CLI and API) ────────────────────────────────

def build_system(verbose: bool = True) -> dict:
    """
    Full pipeline: data → feature store → train 4 models → ranking engine.
    Returns all components for use by API or demo.
    """
    if verbose:
        header("Personalized Product Recommendation System")
        print(f"{C.DIM}  Lenskart / Dyson / Amazon-style multi-layer ML merchandising{C.RESET}")

    # ── Step 1: Generate Data ─────────────────────────────────────────────────
    if verbose:
        step("Step 1/6 — Generating synthetic data")

    from data.generator import generate_all
    t0 = time.perf_counter()
    data = generate_all(n_users=1_000, n_products=500, n_events=50_000, n_experiments=5_000)
    users_df     = data["users"]
    products_df  = data["products"]
    events_df    = data["events"]
    experiments_df = data["experiments"]
    t_data = time.perf_counter() - t0

    if verbose:
        ok(f"Users: {len(users_df):,}  Products: {len(products_df):,}  "
           f"Events: {len(events_df):,}  Experiments: {len(experiments_df):,}  "
           f"({t_data:.2f}s)")

    # ── Step 2: Feature Store ─────────────────────────────────────────────────
    if verbose:
        step("Step 2/6 — Building feature store")

    from data.feature_store import FeatureStore
    t0 = time.perf_counter()
    feature_store = FeatureStore()
    feature_store.build(users_df, products_df, events_df)
    t_fs = time.perf_counter() - t0
    if verbose:
        ok(f"Feature store built in {t_fs:.2f}s — serving {len(users_df):,} users, {len(products_df):,} products")

    # ── Step 3: MLflow tracking setup ─────────────────────────────────────────
    from tracking.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(experiment_name="personalized_merchandising_v1")

    # ── Step 4: Train 4 Models ────────────────────────────────────────────────
    if verbose:
        step("Step 3/6 — Training Model 1: Purchase Propensity")

    from models.propensity import PropensityModel
    propensity_model = PropensityModel()
    with tracker.start_run("propensity_model"):
        tracker.log_params({"n_estimators": 300, "max_depth": 6, "architecture": "XGBoost-TwoTower"})
        prop_metrics = propensity_model.train(events_df, users_df, products_df, verbose=verbose)
        tracker.log_metrics(prop_metrics)
        tracker.log_model_summary("XGBoost propensity model: predicts P(purchase|user,product)")
    if verbose:
        ok(f"Propensity model — Val AUC: {prop_metrics['val_auc']:.4f}")

    if verbose:
        step("Step 4/6 — Training Model 2: Price Elasticity (T-Learner Causal ML)")

    from models.price_elasticity import PriceElasticityModel
    elasticity_model = PriceElasticityModel()
    with tracker.start_run("price_elasticity"):
        tracker.log_params({"architecture": "T-Learner", "treatment": "discount_pct"})
        elast_metrics = elasticity_model.train(experiments_df, products_df, verbose=verbose)
        tracker.log_metrics(elast_metrics)
        tracker.log_model_summary("Causal T-Learner: estimates CATE of price discounts on conversion")

    if verbose:
        ok(f"Elasticity model — MSE(control): {elast_metrics['mse_control']:.6f} | MSE(treated): {elast_metrics['mse_treated']:.6f}")
        step("Step 4b — Training Model 3: Inventory Pressure")

    from models.inventory_pressure import InventoryPressureModel
    inventory_model = InventoryPressureModel()
    with tracker.start_run("inventory_pressure"):
        tracker.log_params({"architecture": "XGBoost", "threshold": 0.65})
        inv_metrics = inventory_model.train(products_df, verbose=verbose)
        tracker.log_metrics(inv_metrics)
        tracker.log_model_summary("XGBoost inventory pressure: deadstock risk [0,1]")
    if verbose:
        ok(f"Inventory model — Val MSE: {inv_metrics['val_mse']:.6f} | AUC(badge): {inv_metrics['val_auc_threshold']:.4f}")

    if verbose:
        step("Step 4c — Training Model 4: Demand Forecast (LightGBM)")

    from models.demand_forecast import DemandForecastModel
    demand_model = DemandForecastModel()
    with tracker.start_run("demand_forecast"):
        tracker.log_params({"architecture": "LightGBM", "horizon_days": 7})
        demand_metrics = demand_model.train(events_df, products_df, verbose=verbose)
        tracker.log_metrics(demand_metrics)
        tracker.log_model_summary("LightGBM demand forecast: 7-day demand spike detection")
    if verbose:
        ok(f"Demand model — MAE: {demand_metrics['mae']:.4f} | RMSE: {demand_metrics['rmse']:.4f}")

    # ── Step 5: Ranking Engine ────────────────────────────────────────────────
    if verbose:
        step("Step 5/6 — Initializing Ranking Engine")

    from ranking.engine import RankingEngine
    t0 = time.perf_counter()
    engine = RankingEngine(
        feature_store=feature_store,
        propensity_model=propensity_model,
        elasticity_model=elasticity_model,
        inventory_model=inventory_model,
        demand_model=demand_model,
        products_df=products_df,
        events_df=events_df,
    )
    t_engine = time.perf_counter() - t0
    if verbose:
        ok(f"Ranking engine ready in {t_engine:.2f}s")

    return {
        "engine": engine,
        "feature_store": feature_store,
        "propensity_model": propensity_model,
        "elasticity_model": elasticity_model,
        "inventory_model": inventory_model,
        "demand_model": demand_model,
        "users_df": users_df,
        "products_df": products_df,
        "events_df": events_df,
        "tracker": tracker,
    }


# ─── Demo ──────────────────────────────────────────────────────────────────────

def run_demo(system: dict) -> None:
    engine          = system["engine"]
    feature_store   = system["feature_store"]
    elasticity_model= system["elasticity_model"]
    inventory_model = system["inventory_model"]
    users_df        = system["users_df"]
    products_df     = system["products_df"]
    events_df       = system["events_df"]
    tracker         = system["tracker"]

    header("Step 6/6 — Live Demo: Personalized Recommendations")

    # Pick 3 sample users
    sample_users = users_df["user_id"].sample(3, random_state=1).tolist()
    presets = ["balanced", "lenskart", "dyson"]

    for uid, preset in zip(sample_users, presets):
        user_row = users_df[users_df["user_id"] == uid].iloc[0]
        print(f"\n{C.BOLD}{C.MAGENTA}{'─'*60}{C.RESET}")
        print(f"  {C.BOLD}User: {uid}{C.RESET}  |  Segment: {user_row['segment']}"
              f"  |  Weight Preset: {C.CYAN}{preset.upper()}{C.RESET}")
        print(f"  Preferred cat: {user_row['preferred_category']}  "
              f"| Price sensitivity: {user_row['price_sensitivity']:.2f}  "
              f"| Sessions (7d): {int(user_row['sessions_7d'])}")
        print(f"{C.BOLD}{C.MAGENTA}{'─'*60}{C.RESET}")

        t0 = time.perf_counter()
        result = engine.recommend(uid, top_k=10, candidate_pool=100, weight_preset=preset)
        latency = (time.perf_counter() - t0) * 1000

        print(f"\n  {'Rank':<5} {'Product':<10} {'Category':<14} {'Brand':<12} "
              f"{'Score':<8} {'Propensity':<12} {'Inv Pressure':<14} {'Trend':<8} "
              f"{'Badge':<8} {'Price'}")
        print(f"  {'-'*110}")

        badge_count = 0
        for p in result.products:
            badge = f"{C.RED}SALE!{C.RESET}" if p.should_show_sale_badge else "      "
            if p.should_show_sale_badge:
                badge_count += 1
            print(
                f"  {p.rank:<5} {p.product_id:<10} {p.category:<14} {p.brand:<12} "
                f"{p.final_score:<8.4f} {p.propensity_score:<12.4f} {p.inventory_pressure:<14.4f} "
                f"{p.trend_score:<8.4f} {badge:<8} ${p.current_price:.2f}"
            )

        print(f"\n  {C.GREEN}Latency: {latency:.1f}ms{C.RESET}  |  "
              f"Candidates scored: {result.total_candidates}  |  "
              f"Sale badges triggered: {badge_count}/{len(result.products)}")

    # ── Price Elasticity Demo ─────────────────────────────────────────────────
    header("Price Elasticity — Optimal Discount per Product")
    sample_pids = products_df["product_id"].sample(5, random_state=42).tolist()
    print(f"\n  {'Product':<12} {'Category':<14} {'Base Price':<12} "
          f"{'Opt Discount':<14} {'Expected Uplift':<18} {'Urgency'}")
    print(f"  {'-'*80}")

    for pid in sample_pids:
        disc_info = elasticity_model.optimal_discount(pid)
        prod_row = products_df[products_df["product_id"] == pid].iloc[0]
        inv_score = engine._inventory_scores.get(pid, 0.5)
        urgency = inventory_model.urgency_label(inv_score)
        urgency_colored = {
            "Critical": f"{C.RED}Critical{C.RESET}",
            "High":     f"{C.YELLOW}High{C.RESET}",
            "Medium":   f"{C.CYAN}Medium{C.RESET}",
            "Low":      f"{C.GREEN}Low{C.RESET}",
        }.get(urgency, urgency)

        print(
            f"  {pid:<12} {prod_row['category']:<14} ${prod_row['base_price']:<10.2f} "
            f"{disc_info['optimal_discount_pct']*100:.0f}%{'':<11} "
            f"+{disc_info['expected_uplift']*100:.1f}%{'':<14} "
            f"{urgency_colored}"
        )

    # ── A/B Variant Assignment Demo ───────────────────────────────────────────
    header("A/B Test Variant Assignment")
    from ab_testing.feature_flags import assign_ab_variant
    print(f"\n  {'User':<12} {'Variant':<12} {'Description'}")
    print(f"  {'-'*60}")
    for uid in sample_users:
        variant = assign_ab_variant(uid)
        from ab_testing.feature_flags import get_weight_config
        cfg = get_weight_config(variant)
        print(f"  {uid:<12} {C.CYAN}{variant:<12}{C.RESET} {cfg.description}")

    # ── Drift Detection Demo ──────────────────────────────────────────────────
    header("Drift Detection — Monitoring Feature Stability")
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector()

    # Use user features as the thing to monitor
    user_numeric = users_df.select_dtypes(include=[np.number]).head(600)
    detector.fit_reference(user_numeric)

    # Simulate "drifted" incoming data: slightly shift distributions
    rng = np.random.default_rng(99)
    drifted = user_numeric.copy()
    drifted["sessions_7d"] = drifted["sessions_7d"] * 2 + rng.normal(0, 2, len(drifted))
    drifted["price_sensitivity"] = rng.uniform(0.6, 1.0, len(drifted))

    print(f"\n  Checking {len(drifted)} samples against reference distribution...\n")
    alerts = detector.check(drifted.head(400), verbose=False)

    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"  {C.GREEN}No drift detected{C.RESET}")

    print(f"\n  Monitoring {detector.summary()['n_numeric_features']} numeric features | "
          f"PSI threshold: warning={detector.summary()['psi_thresholds']['warning']}, "
          f"critical={detector.summary()['psi_thresholds']['critical']}")

    # ── MLflow Summary ────────────────────────────────────────────────────────
    tracker.print_summary()

    # ── Final Banner ──────────────────────────────────────────────────────────
    header("System Ready")
    print(f"""
  {C.BOLD}Components built:{C.RESET}
    {C.GREEN}✓{C.RESET}  Model 1: Purchase Propensity          (XGBoost, AUC={system['tracker']._runs[0]['metrics'].get('val_auc', '?')})
    {C.GREEN}✓{C.RESET}  Model 2: Price Elasticity             (T-Learner Causal ML)
    {C.GREEN}✓{C.RESET}  Model 3: Inventory Pressure           (XGBoost deadstock detector)
    {C.GREEN}✓{C.RESET}  Model 4: Demand Forecast              (LightGBM, 7-day horizon)
    {C.GREEN}✓{C.RESET}  Ranking Engine                        (weighted scoring + MMR diversity)
    {C.GREEN}✓{C.RESET}  Feature Store                         (in-memory, O(1) serving)
    {C.GREEN}✓{C.RESET}  A/B Testing                           (5 weight presets, hash assignment)
    {C.GREEN}✓{C.RESET}  Drift Detector                        (PSI + chi-square + z-score)
    {C.GREEN}✓{C.RESET}  MLflow Tracker                        (logs at ./mlruns/)
    {C.GREEN}✓{C.RESET}  FastAPI Server                        (run with --serve flag)

  {C.BOLD}To start the API server:{C.RESET}
    {C.CYAN}python file.py --serve{C.RESET}

  {C.BOLD}API endpoints (once running):{C.RESET}
    {C.DIM}POST  http://localhost:8080/recommend
    POST  http://localhost:8080/score
    GET   http://localhost:8080/sale-badge/{{product_id}}
    GET   http://localhost:8080/weight-presets
    GET   http://localhost:8080/health
    GET   http://localhost:8080/docs   (Swagger UI){C.RESET}
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Personalized Product Recommendation + Dynamic Merchandising System"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start FastAPI server on port 8080 after pipeline demo"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for FastAPI server (default: 8080)"
    )
    args = parser.parse_args()

    # Build system and run demo
    system = build_system(verbose=True)
    run_demo(system)

    # Optionally start API server
    if args.serve:
        header(f"Starting FastAPI Server on port {args.port}")
        print(f"  {C.CYAN}Swagger docs → http://localhost:{args.port}/docs{C.RESET}\n")

        # Store system globally for API to reuse
        import api.app as api_module
        api_module._engine          = system["engine"]
        api_module._feature_store   = system["feature_store"]
        api_module._inventory_model = system["inventory_model"]
        api_module._elasticity_model= system["elasticity_model"]

        import uvicorn
        uvicorn.run(
            api_module.app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
