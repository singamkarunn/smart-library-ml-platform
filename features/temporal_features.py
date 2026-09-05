"""
features/temporal_features.py
------------------------------
Extracts time-based features from LMS borrowing records.

Temporal features capture WHEN patrons borrow — day of week,
time of month, seasonality, borrowing frequency over time windows.
These are often the strongest signals in recommendation systems
because patron behavior follows strong weekly and seasonal patterns.

Output: A patron-level DataFrame where each row is one patron
        and each column is a temporal feature ready for modeling.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Core Temporal Feature Extractor ──────────────────────────────────────
def extract_temporal_features(lms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal features from LMS borrowing records.

    Features engineered:
    - Borrowing frequency over 30, 60, 90 day windows
    - Preferred day of week (most common checkout day)
    - Preferred month (most common checkout month)
    - Season preference (spring/summer/fall/winter)
    - Average days between checkouts (borrowing cadence)
    - Recency: days since last checkout
    - Tenure: days between first and last checkout
    - Weekend vs weekday borrowing ratio
    - Peak hour preference (morning/afternoon/evening)

    Args:
        lms_df: Raw LMS borrowing records from lms_connector

    Returns:
        pd.DataFrame with one row per patron and temporal feature columns
    """
    logger.info("Extracting temporal features from LMS data...")

    df = lms_df.copy()

    # Ensure datetime types
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])
    df["return_date"] = pd.to_datetime(df["return_date"])

    # Reference date — in production this would be today
    # In development we use the max date in the dataset
    reference_date = df["checkout_date"].max()

    # ── Day / Month / Season helpers ──────────────────────────────────────
    df["checkout_dayofweek"] = df["checkout_date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["checkout_month"] = df["checkout_date"].dt.month
    df["checkout_hour"] = df["checkout_date"].dt.hour
    df["is_weekend"] = df["checkout_dayofweek"].isin([5, 6]).astype(int)

    def get_season(month):
        if month in [12, 1, 2]:  return "winter"
        elif month in [3, 4, 5]: return "spring"
        elif month in [6, 7, 8]: return "summer"
        else:                     return "fall"

    df["season"] = df["checkout_month"].apply(get_season)

    def get_time_of_day(hour):
        if hour < 12:   return "morning"
        elif hour < 17: return "afternoon"
        else:           return "evening"

    df["time_of_day"] = df["checkout_hour"].apply(get_time_of_day)

    # ── Per-patron aggregations ───────────────────────────────────────────
    patron_groups = df.groupby("patron_id")

    features = pd.DataFrame(index=df["patron_id"].unique())
    features.index.name = "patron_id"

    # Total checkouts
    features["total_checkouts"] = patron_groups["book_id"].count()

    # Recency — days since last checkout
    features["days_since_last_checkout"] = (
        reference_date - patron_groups["checkout_date"].max()
    ).dt.days

    # Tenure — days between first and last checkout
    features["tenure_days"] = (
        patron_groups["checkout_date"].max() -
        patron_groups["checkout_date"].min()
    ).dt.days

    # Average loan duration
    features["avg_loan_duration_days"] = patron_groups["loan_duration_days"].mean().round(2)

    # Renewal rate
    features["avg_renewals_per_checkout"] = patron_groups["times_renewed"].mean().round(3)

    # Weekend borrowing ratio
    weekend_counts = df[df["is_weekend"] == 1].groupby("patron_id")["book_id"].count()
    features["weekend_borrow_ratio"] = (
        weekend_counts / features["total_checkouts"]
    ).fillna(0).round(3)

    # Preferred day of week (mode)
    features["preferred_day_of_week"] = patron_groups["checkout_dayofweek"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else -1
    )

    # Preferred month (mode)
    features["preferred_month"] = patron_groups["checkout_month"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else -1
    )

    # Preferred season (mode)
    features["preferred_season"] = patron_groups["season"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "unknown"
    )

    # Preferred time of day (mode)
    features["preferred_time_of_day"] = patron_groups["time_of_day"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "unknown"
    )

    # ── Rolling window features ───────────────────────────────────────────
    # How active has the patron been recently vs overall?
    # This captures engagement trends — a patron ramping up vs cooling off

    for window_days in [30, 60, 90]:
        cutoff = reference_date - pd.Timedelta(days=window_days)
        recent = df[df["checkout_date"] >= cutoff].groupby("patron_id")["book_id"].count()
        features[f"checkouts_last_{window_days}d"] = recent.fillna(0).astype(int)

    # Borrowing velocity — checkouts per month of tenure
    features["monthly_borrow_rate"] = (
        features["total_checkouts"] /
        (features["tenure_days"] / 30).replace(0, 1)
    ).round(3)

    # Overdue rate
    overdue_counts = df[df["loan_status"] == "overdue"].groupby("patron_id")["book_id"].count()
    features["overdue_rate"] = (
        overdue_counts / features["total_checkouts"]
    ).fillna(0).round(3)

    features = features.reset_index()
    logger.info(
        f"Extracted {len(features.columns) - 1} temporal features "
        f"for {len(features)} patrons"
    )
    return features


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python features/temporal_features.py
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingestion"))

    from lms_connector import load_lms_data

    print("Loading LMS data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=5000)

    print("Extracting temporal features...")
    features_df = extract_temporal_features(lms_df)

    print("\n── Sample Features ──")
    print(features_df.head())
    print("\n── Feature Columns ──")
    print(features_df.columns.tolist())
    print(f"\nShape: {features_df.shape}")
    print("\n── Feature Stats ──")
    print(features_df.describe().round(2))