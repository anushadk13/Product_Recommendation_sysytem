"""
data/generator.py
─────────────────
Synthetic data generator for the personalized merchandising system.

Generates:
  - Users          : 1 000 users with behavioral profiles
  - Products       : 500 SKUs with category / price / inventory attributes
  - Purchase events: 50 000 historical transactions with seasonality
  - Price experiments: random price-drop experiments for elasticity training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


# ─── Constants ────────────────────────────────────────────────────────────────
CATEGORIES = ["Eyewear", "Electronics", "Home", "Fashion", "Beauty", "Sports"]
BRANDS = ["LensX", "OptiPro", "SkyTech", "HomeCore", "StyleZ", "ActiveFit"]
SEED = 42


def generate_users(n_users: int = 1_000, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic user profiles with behavioral features."""
    rng = np.random.default_rng(seed)
    now = datetime.utcnow()

    user_ids = [f"U{str(i).zfill(5)}" for i in range(n_users)]

    df = pd.DataFrame({
        "user_id": user_ids,
        # Recency: days since last purchase (lower = more active)
        "recency_days": rng.integers(1, 365, size=n_users),
        # Frequency: purchases in last 90 days
        "purchase_freq_90d": rng.integers(0, 20, size=n_users),
        # Average order value
        "avg_order_value": rng.uniform(20, 500, size=n_users).round(2),
        # Total lifetime spend
        "lifetime_spend": rng.uniform(50, 10_000, size=n_users).round(2),
        # Category affinity (one-hot encoded below)
        "preferred_category": rng.choice(CATEGORIES, size=n_users),
        # Device type
        "device_type": rng.choice(["mobile", "desktop", "tablet"], size=n_users, p=[0.6, 0.3, 0.1]),
        # Price sensitivity score (0=insensitive, 1=very sensitive)
        "price_sensitivity": rng.uniform(0, 1, size=n_users).round(3),
        # Days since account creation
        "account_age_days": rng.integers(30, 1825, size=n_users),
        # Session count last 7 days
        "sessions_7d": rng.integers(0, 15, size=n_users),
        # Cart abandonment rate
        "cart_abandon_rate": rng.uniform(0, 1, size=n_users).round(3),
        # Customer segment
        "segment": rng.choice(["VIP", "Regular", "New", "Lapsed"], size=n_users, p=[0.1, 0.5, 0.25, 0.15]),
    })

    # Add category affinity scores (soft encoding)
    for cat in CATEGORIES:
        df[f"affinity_{cat.lower()}"] = rng.uniform(0, 1, size=n_users).round(3)
    # Boost preferred category affinity
    for idx, row in df.iterrows():
        cat_col = f"affinity_{row['preferred_category'].lower()}"
        df.at[idx, cat_col] = min(1.0, df.at[idx, cat_col] + 0.4)

    return df


def generate_products(n_products: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic product catalogue with inventory and pricing features."""
    rng = np.random.default_rng(seed + 1)

    product_ids = [f"P{str(i).zfill(5)}" for i in range(n_products)]
    categories = rng.choice(CATEGORIES, size=n_products)
    brands = rng.choice(BRANDS, size=n_products)

    base_prices = rng.uniform(15, 800, size=n_products).round(2)
    cost_prices = (base_prices * rng.uniform(0.3, 0.6, size=n_products)).round(2)
    current_stock = rng.integers(0, 500, size=n_products)

    df = pd.DataFrame({
        "product_id": product_ids,
        "category": categories,
        "brand": brands,
        "base_price": base_prices,
        "current_price": (base_prices * rng.uniform(0.7, 1.0, size=n_products)).round(2),
        "cost_price": cost_prices,
        # Inventory
        "current_stock": current_stock,
        "initial_stock": current_stock + rng.integers(0, 300, size=n_products),
        "reorder_threshold": rng.integers(5, 50, size=n_products),
        "days_on_shelf": rng.integers(1, 180, size=n_products),
        # Ratings
        "avg_rating": rng.uniform(2.5, 5.0, size=n_products).round(1),
        "review_count": rng.integers(0, 5_000, size=n_products),
        # Trend (rolling 7d views normalized)
        "trend_score": rng.uniform(0, 1, size=n_products).round(3),
        # Is new product launch (< 30 days)
        "is_new_launch": rng.integers(0, 2, size=n_products),
        # Weight for diversity penalty (embedding cluster)
        "style_cluster": rng.integers(0, 20, size=n_products),
        # Seasonal relevance (0-1)
        "seasonal_relevance": rng.uniform(0, 1, size=n_products).round(3),
    })

    # Sell-through rate = units sold / initial stock
    df["sell_through_rate"] = (
        (df["initial_stock"] - df["current_stock"]) / df["initial_stock"].clip(lower=1)
    ).clip(0, 1).round(3)

    # Margin score = (price - cost) / price
    df["margin_score"] = (
        (df["current_price"] - df["cost_price"]) / df["current_price"].clip(lower=0.01)
    ).clip(0, 1).round(3)

    return df


def generate_purchase_events(
    users: pd.DataFrame,
    products: pd.DataFrame,
    n_events: int = 50_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate historical purchase event log with seasonality."""
    rng = np.random.default_rng(seed + 2)
    now = datetime.utcnow()

    user_ids = users["user_id"].values
    product_ids = products["product_id"].values

    # Bias toward high-affinity category matches
    sampled_users = rng.choice(user_ids, size=n_events)
    sampled_products = rng.choice(product_ids, size=n_events)

    # Random timestamps over last 365 days with weekly seasonality
    days_ago = rng.integers(0, 365, size=n_events)
    hours = rng.integers(0, 24, size=n_events)
    timestamps = [now - timedelta(days=int(d), hours=int(h)) for d, h in zip(days_ago, hours)]

    prices_paid = []
    for pid in sampled_products:
        row = products[products["product_id"] == pid].iloc[0]
        # Add some price variation (simulating experiments)
        discount = rng.uniform(0.7, 1.0)
        prices_paid.append(round(float(row["current_price"]) * discount, 2))

    df = pd.DataFrame({
        "event_id": [f"E{str(i).zfill(7)}" for i in range(n_events)],
        "user_id": sampled_users,
        "product_id": sampled_products,
        "timestamp": timestamps,
        "price_paid": prices_paid,
        "quantity": rng.integers(1, 4, size=n_events),
        "event_type": rng.choice(
            ["purchase", "add_to_cart", "wishlist", "view"],
            size=n_events,
            p=[0.25, 0.20, 0.10, 0.45],
        ),
        # Device at time of event
        "device": rng.choice(["mobile", "desktop", "tablet"], size=n_events, p=[0.6, 0.3, 0.1]),
    })

    return df


def generate_price_experiments(
    products: pd.DataFrame,
    n_experiments: int = 5_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Simulate price A/B experiments for elasticity training.
    Returns rows with (product_id, discount_pct, conversion_rate).
    """
    rng = np.random.default_rng(seed + 3)
    product_ids = products["product_id"].values
    sampled_products = rng.choice(product_ids, size=n_experiments)

    discount_pcts = rng.uniform(0, 0.5, size=n_experiments).round(3)

    # Simulate conversion: higher discount → higher conversion, modulated by price sensitivity
    base_conversion = rng.uniform(0.01, 0.15, size=n_experiments)
    elasticity_coeff = rng.uniform(0.5, 3.0, size=n_experiments)
    conversion_rates = (base_conversion + elasticity_coeff * discount_pcts).clip(0, 1).round(4)

    # Add noise
    conversion_rates += rng.normal(0, 0.01, size=n_experiments)
    conversion_rates = conversion_rates.clip(0, 1).round(4)

    return pd.DataFrame({
        "product_id": sampled_products,
        "discount_pct": discount_pcts,
        "conversion_rate": conversion_rates,
        "elasticity_coeff": elasticity_coeff.round(3),
        "n_users_in_experiment": rng.integers(100, 5000, size=n_experiments),
    })


def generate_all(
    n_users: int = 1_000,
    n_products: int = 500,
    n_events: int = 50_000,
    n_experiments: int = 5_000,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset."""
    print("[DataGen] Generating users...")
    users = generate_users(n_users, seed)
    print(f"  → {len(users)} users")

    print("[DataGen] Generating products...")
    products = generate_products(n_products, seed)
    print(f"  → {len(products)} products")

    print("[DataGen] Generating purchase events...")
    events = generate_purchase_events(users, products, n_events, seed)
    print(f"  → {len(events)} events")

    print("[DataGen] Generating price experiments...")
    experiments = generate_price_experiments(products, n_experiments, seed)
    print(f"  → {len(experiments)} experiment rows")

    return {
        "users": users,
        "products": products,
        "events": events,
        "experiments": experiments,
    }


if __name__ == "__main__":
    data = generate_all()
    for name, df in data.items():
        print(f"\n{name}.head(3):")
        print(df.head(3).to_string())
