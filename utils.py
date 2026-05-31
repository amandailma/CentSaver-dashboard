"""
utils.py
Utility functions for CentSaver Streamlit Dashboard.
Handles data loading, feature engineering (anti-leakage), and preprocessing.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(uploaded_file):
    """Load and perform initial cleaning."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "category"])
    df = df[df["amount"] > 0].copy()
    return df

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Anti-Leakage)
# ---------------------------------------------------------------------------
def engineer_features(df):
    """
    Create all features needed for EDA and model inference.
    Uses category-aware baseline to avoid leakage.
    """
    df = df.copy()

    # Temporal features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Senin
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_type"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")
    df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["amount_log"] = np.log1p(df["amount"])

    # Category-aware baseline (computed from historical data = entire uploaded set)
    category_day_stats = (
        df.groupby(["category", "day_type"])
        .agg(
            cat_day_avg=("amount", "mean"),
            cat_day_median=("amount", "median"),
            cat_day_std=("amount", "std"),
            cat_day_q25=("amount", lambda x: x.quantile(0.25)),
            cat_day_q75=("amount", lambda x: x.quantile(0.75)),
            cat_day_count=("amount", "size"),
        )
        .reset_index()
    )

    df = df.merge(category_day_stats, on=["category", "day_type"], how="left")
    df["cat_day_std"] = df["cat_day_std"].fillna(df["amount"].std())

    # Amount ratio & zscore (for EDA/microspending flag ONLY, NOT model input)
    df["amount_ratio"] = df["amount"] / df["cat_day_avg"].replace(0, np.nan)
    df["amount_ratio"] = df["amount_ratio"].fillna(1.0)
    df["amount_zscore"] = (df["amount"] - df["cat_day_avg"]) / df["cat_day_std"].replace(0, np.nan)
    df["amount_zscore"] = df["amount_zscore"].fillna(0)

    # Flags
    df["small_amount_flag"] = (
        (df["amount"] <= df["cat_day_q25"]) & 
        (df["amount"] <= df["cat_day_median"])
    ).astype(int)

    # Frequency baseline per category
    monthly_counts = df.groupby(["period", "category"]).size().reset_index(name="monthly_txn_count")
    freq_stats = monthly_counts.groupby("category").agg(
        monthly_txn_count_avg=("monthly_txn_count", "mean"),
        monthly_txn_count_median=("monthly_txn_count", "median"),
    ).reset_index()

    df = df.merge(freq_stats, on="category", how="left")
    global_freq_median = monthly_counts["monthly_txn_count"].median()
    df["monthly_txn_count_median"] = df["monthly_txn_count_median"].fillna(global_freq_median)

    df["repetitive_category_flag"] = (
        df["monthly_txn_count_avg"] >= df["monthly_txn_count_median"]
    ).astype(int)

    # Adaptive microspending label (for display/EDA)
    df["is_adaptive_microspending"] = (
        (df["small_amount_flag"] == 1) & 
        (df["repetitive_category_flag"] == 1)
    ).astype(int)

    # Category encoding for model
    cat_map = {c: i+1 for i, c in enumerate(sorted(df["category"].unique()))}
    df["cat_encoded"] = df["category"].map(cat_map)

    return df

# ---------------------------------------------------------------------------
# 3. RFM-STYLE SEGMENTATION
# ---------------------------------------------------------------------------
def compute_rfm(df):
    """Compute user-level RFM-style segmentation."""
    user_rfm = (
        df.groupby("description")
        .agg(
            recency=("date", lambda x: (df["date"].max() - x.max()).days),
            frequency=("amount", "size"),
            monetary=("amount", "sum"),
            avg_amount=("amount", "mean"),
            micro_count=("label", lambda x: (x == 1).sum()),
            category_mode=("category", lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        )
        .reset_index()
    )
    user_rfm["micro_rate"] = user_rfm["micro_count"] / user_rfm["frequency"]

    # Monetary quartile
    user_rfm["monetary_segment"] = pd.qcut(
        user_rfm["monetary"], q=4, labels=["Low", "Medium", "High", "Premium"], duplicates="drop"
    )

    # Frequency quartile (handle edge cases)
    n_unique = user_rfm["frequency"].nunique()
    labels = ["Rare", "Occasional", "Regular", "Frequent"][:min(n_unique, 4)]
    if n_unique >= 2:
        user_rfm["frequency_segment"] = pd.qcut(
            user_rfm["frequency"], q=len(labels), labels=labels, duplicates="drop"
        )
    else:
        user_rfm["frequency_segment"] = "Single-Frequency"

    return user_rfm

# ---------------------------------------------------------------------------
# 4. MICROSPENDING RATIO (Quest #1)
# ---------------------------------------------------------------------------
def compute_microspending_ratio(df):
    """Compute monthly microspending ratio per category and overall."""
    monthly_micro = (
        df.groupby(["period", "category"])
        .agg(
            total_monthly=("amount", "sum"),
            micro_monthly=("amount", lambda x: x[df.loc[x.index, "is_adaptive_microspending"] == 1].sum()),
            txn_count=("amount", "size"),
            micro_count=("is_adaptive_microspending", "sum")
        )
        .reset_index()
    )
    monthly_micro["micro_pct"] = (monthly_micro["micro_monthly"] / monthly_micro["total_monthly"] * 100).fillna(0)
    monthly_micro["micro_pct"] = monthly_micro["micro_pct"].clip(0, 100)

    # Overall monthly
    overall = (
        df.groupby("period")
        .agg(
            total_monthly=("amount", "sum"),
            micro_monthly=("amount", lambda x: x[df.loc[x.index, "is_adaptive_microspending"] == 1].sum())
        )
        .reset_index()
    )
    overall["micro_pct"] = (overall["micro_monthly"] / overall["total_monthly"] * 100).fillna(0)

    # Average per category
    avg_by_cat = (
        monthly_micro.groupby("category")
        .agg(avg_micro_pct=("micro_pct", "mean"))
        .sort_values("avg_micro_pct", ascending=False)
        .reset_index()
    )

    return monthly_micro, overall, avg_by_cat

# ---------------------------------------------------------------------------
# 5. ANOMALY & MoM GROWTH (Quest #3)
# ---------------------------------------------------------------------------
def compute_mom_and_anomaly(df):
    """Compute Month-over-Month growth and anomaly flags per category."""
    monthly_cat = (
        df.groupby(["period", "category"])
        .agg(total_amount=("amount", "sum"), txn_count=("amount", "size"))
        .reset_index()
    )
    monthly_cat = monthly_cat.sort_values(["category", "period"])
    monthly_cat["prev_amount"] = monthly_cat.groupby("category")["total_amount"].shift(1)
    monthly_cat["mom_growth"] = (
        (monthly_cat["total_amount"] - monthly_cat["prev_amount"]) / 
        (monthly_cat["prev_amount"] + 1e-9)
    ).replace([np.inf, -np.inf], 0)
    monthly_cat["mom_growth_pct"] = monthly_cat["mom_growth"] * 100

    # Z-score per category
    monthly_cat["amount_zscore"] = monthly_cat.groupby("category")["total_amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    monthly_cat["is_anomaly"] = (monthly_cat["amount_zscore"].abs() > 2).astype(int)

    # Anomaly frequency
    anomaly_freq = (
        monthly_cat.groupby("category")
        .agg(
            anomaly_count=("is_anomaly", "sum"),
            total_months=("period", "size"),
            avg_zscore=("amount_zscore", lambda x: x.abs().mean()),
            max_growth=("mom_growth_pct", "max"),
        )
        .reset_index()
    )
    anomaly_freq["anomaly_rate"] = anomaly_freq["anomaly_count"] / anomaly_freq["total_months"]
    anomaly_freq = anomaly_freq.sort_values("anomaly_rate", ascending=False)

    # Weekend impulse
    weekend_impulse = (
        df.groupby(["category", "is_weekend"])
        .agg(risk_rate=("label", "mean"))
        .reset_index()
        .pivot(index="category", columns="is_weekend", values="risk_rate")
        .fillna(0)
    )
    weekend_impulse["weekend_boost"] = weekend_impulse.get(1, 0) - weekend_impulse.get(0, 0)
    weekend_impulse = weekend_impulse.sort_values("weekend_boost", ascending=False).head(8)

    return monthly_cat, anomaly_freq, weekend_impulse

# ---------------------------------------------------------------------------
# 6. MODEL INPUT PREPARATION
# ---------------------------------------------------------------------------
def prepare_model_input(df):
    """Prepare feature matrix X for model inference (anti-leakage)."""
    features = [
        "amount", "amount_log", "day_of_week", "is_weekend",
        "month", "year", "cat_encoded",
        "cat_day_q25", "cat_day_median", "cat_day_count",
        "monthly_txn_count_avg", "monthly_txn_count_median",
        "small_amount_flag", "repetitive_category_flag"
    ]
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0)
    return X, available
