# Personalized Product Recommendation & Dynamic Merchandising System

A production-grade, multi-layer Machine Learning system modeled after the real-time personalized merchandising engines used by companies like Lenskart, Dyson, and Amazon.
Instead of a simple "should this go on sale?" binary, this system runs 4 ML models simultaneously and uses a weighted ranking engine to generate a personalized product grid in real-time (sub-50ms latency).

# The 4 Core ML Models

## Purchase Propensity (XGBoost Two-Tower)

Question: Will this user buy this product in the next 24 hours?
Features: User behavioral stats (recency, frequency, spend) crossed with product attributes.

## Price Elasticity (Causal ML T-Learner)

Question: If we drop this product's price by 10%, how much does conversion jump?
Output: Calculates the Optimal Discount percentage to maximize conversion lift against margin loss.

## Inventory Pressure (XGBoost Regressor)

Question: Is this SKU at risk of becoming deadstock?
Features: Stock ratio, sell-through rate, days on shelf.
Action: Triggers a "SALE" badge on the frontend if pressure is high.

## Demand Forecast (LightGBM Time-Series)

Question: Will demand spike for this product in the next 7 days?
Features: Rolling 7d/14d/30d purchase velocity.
Action: Triggers a "Trending" badge on the frontend to pre-position stock before peaks.

## The Ranking Engine

Rather than a simple classification, the system treats personalization as a ranking problem. It calculates a final_score for every (user, product) pair using A/B-testable weights:

final_score = (
    W_propensity  * propensity_score
  + W_inventory   * inventory_pressure_score
  + W_margin      * margin_score
  + W_trend       * trend_score
  + W_demand      * demand_spike_score
)
Maximal Marginal Relevance (MMR) Diversity Re-ranking is applied at the end to penalize similar items, preventing the engine from recommending 10 identical products.


## Architecture Stack

Backend: Python, FastAPI, XGBoost, LightGBM, Pandas, Scikit-Learn
Tracking: MLflow (SQLite backend)
Feature Serving: Custom In-Memory Feature Store (O(1) latency)
Frontend: Vite, Vanilla JavaScript, CSS Glassmorphism
Monitoring: Custom Feature Drift Detector (PSI, Chi-Square, Z-Score)

# How to Run the Project
You will need two terminal windows to run the backend API and the frontend dashboard simultaneously.

1. Start the Backend API
The backend script will automatically generate 50,000 rows of synthetic mock data, build the feature store, train all 4 ML models via MLflow, and start the FastAPI server on port 8080.

# Open Terminal 1
python file.py --serve
You can view the API documentation at http://localhost:8080/docs

2. Start the Frontend Dashboard
Once the backend says "server ready", open a new terminal window to start the Vite UI.

# Open Terminal 2
npm install    # (only needed the first time)
npm run dev -- --port 5173 --host

3. View the Dashboard
Open your web browser and navigate to: 👉 http://localhost:5173/

From the dashboard you can:
Change the User Context to see personalization update instantly.
Change the A/B Strategy (e.g., Lenskart clearance vs Amazon max conversion) to watch the ranking weights shift.
Click on any product card to see the full mathematical breakdown of its score across all 4 models.

<img src ="https://github.com/anushadk13/Product_Recommendation_sysytem/blob/main/frontend/public/Screenshot%202026-06-29%20at%209.53.43%E2%80%AFAM.png">
