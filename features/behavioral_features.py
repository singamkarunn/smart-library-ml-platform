"""
features/behavioral_features.py
--------------------------------
Extracts patron behavior features from LMS, POS, and activity data.

Behavioral features capture HOW patrons interact with the library —
genre preferences, spending patterns, digital engagement, loyalty signals.
Combined with temporal features, these form the core input to the
collaborative filtering and hybrid recommendation models.

Output: A patron-level DataFrame with behavioral feature columns
        ready to be joined with temporal features for model training.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Genre Preference Features ─────────────────────────────────────────────
def extract_genre_features(lms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts genre preference features from borrowing history.

    For each patron we calculate:
    - Top genre (most borrowed)
    - Genre diversity (number of unique genres borrowed)
    - Genre concentration (what % of borrows is top genre)
    - Per-genre borrowing counts (one column per genre)

    Args:
        lms_df: Raw LMS borrowing records

    Returns:
        pd.DataFrame with one row per patron and genre feature columns
    """
    logger.info("Extracting genre preference features...")

    # Genre counts per patron
    genre_counts = lms_df.groupby(
        ["patron_id", "genre"]
    )["book_id"].count().unstack(fill_value=0)

    # Prefix columns so they're identifiable downstream
    genre_counts.columns = [f"genre_{g.lower().replace(' ', '_')}_count"
                             for g in genre_counts.columns]

    features = genre_counts.copy()

    # Total borrows (denominator for ratios)
    total = genre_counts.sum(axis=1)

    # Genre diversity — number of genres borrowed at least once
    features["genre_diversity"] = (genre_counts > 0).sum(axis=1)

    # Top genre
    raw_genres = lms_df.groupby(["patron_id", "genre"])["book_id"].count().unstack(fill_value=0)
    features["top_genre"] = raw_genres.idxmax(axis=1)

    # Genre concentration — how dominant is the top genre?
    features["genre_concentration"] = (
        raw_genres.max(axis=1) / total
    ).round(3)

    features = features.reset_index()
    features.rename(columns={"patron_id": "patron_id"}, inplace=True)

    logger.info(f"Genre features extracted for {len(features)} patrons")
    return features


# ── POS Spending Features ─────────────────────────────────────────────────
def extract_spending_features(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts spending behavior features from POS transaction data.

    Spending patterns are strong engagement signals — patrons who
    spend at the cafe and buy merchandise are more invested in the
    library experience and tend to borrow more consistently.

    Args:
        pos_df: Raw POS transaction records

    Returns:
        pd.DataFrame with one row per patron and spending feature columns
    """
    logger.info("Extracting spending behavior features...")

    patron_groups = pos_df.groupby("patron_id")

    features = pd.DataFrame()
    features["patron_id"] = pos_df["patron_id"].unique()
    features = features.set_index("patron_id")

    # Total spend and transaction count
    features["total_spend"] = patron_groups["amount"].sum().round(2)
    features["total_pos_transactions"] = patron_groups["amount"].count()
    features["avg_transaction_value"] = patron_groups["amount"].mean().round(2)
    features["max_single_transaction"] = patron_groups["amount"].max().round(2)

    # Spend by category
    for txn_type in ["cafe", "merchandise", "printing", "event_ticket", "donation"]:
        type_spend = pos_df[pos_df["transaction_type"] == txn_type].groupby(
            "patron_id"
        )["amount"].sum()
        features[f"spend_{txn_type}"] = type_spend.fillna(0).round(2)

    # Event ticket buyer flag — strong engagement signal
    event_buyers = pos_df[pos_df["transaction_type"] == "event_ticket"]["patron_id"].unique()
    features["is_event_buyer"] = features.index.isin(event_buyers).astype(int)

    # Donor flag
    donors = pos_df[pos_df["transaction_type"] == "donation"]["patron_id"].unique()
    features["is_donor"] = features.index.isin(donors).astype(int)

    # Preferred payment method
    features["preferred_payment"] = patron_groups["payment_method"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "unknown"
    )

    features = features.reset_index()
    logger.info(f"Spending features extracted for {len(features)} patrons")
    return features


# ── Activity Engagement Features ──────────────────────────────────────────
def extract_engagement_features(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts digital engagement features from patron activity events.

    Activity events reveal search behavior, wishlist usage, and
    session patterns — signals that predict what a patron wants
    before they actually borrow it.

    Args:
        activity_df: Raw patron activity events from kafka_consumer

    Returns:
        pd.DataFrame with one row per patron and engagement feature columns
    """
    logger.info("Extracting engagement features from activity events...")

    patron_groups = activity_df.groupby("patron_id")

    features = pd.DataFrame()
    features["patron_id"] = activity_df["patron_id"].unique()
    features = features.set_index("patron_id")

    # Total events
    features["total_events"] = patron_groups["event_id"].count()

    # Per event type counts
    for event_type in ["book_search", "book_view", "wishlist_add",
                        "reservation_made", "review_submitted"]:
        event_counts = activity_df[
            activity_df["event_type"] == event_type
        ].groupby("patron_id")["event_id"].count()
        features[f"count_{event_type}"] = event_counts.fillna(0).astype(int)

    # Wishlist conversion rate — how often does a view lead to a wishlist add?
    features["wishlist_conversion_rate"] = (
        features["count_wishlist_add"] /
        features["count_book_view"].replace(0, 1)
    ).round(3)

    # Reservation rate — how often does a search lead to a reservation?
    features["reservation_rate"] = (
        features["count_reservation_made"] /
        features["count_book_search"].replace(0, 1)
    ).round(3)

    # Review engagement — did they ever submit a review?
    features["is_reviewer"] = (features["count_review_submitted"] > 0).astype(int)

    # Device preference
    features["preferred_device"] = patron_groups["device_type"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "unknown"
    )

    # Session count
    session_counts = activity_df[
        activity_df["event_type"] == "session_start"
    ].groupby("patron_id")["event_id"].count()
    features["total_sessions"] = session_counts.fillna(0).astype(int)

    # Events per session — engagement depth
    features["events_per_session"] = (
        features["total_events"] /
        features["total_sessions"].replace(0, 1)
    ).round(2)

    features = features.reset_index()
    logger.info(f"Engagement features extracted for {len(features)} patrons")
    return features


# ── Master Behavioral Feature Builder ─────────────────────────────────────
def build_behavioral_features(lms_df: pd.DataFrame,
                               pos_df: pd.DataFrame,
                               activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all behavioral feature sets into a single patron-level DataFrame.

    Joins genre, spending, and engagement features on patron_id.
    Missing values are filled with 0 for counts, "unknown" for categories.

    Args:
        lms_df:      LMS borrowing records
        pos_df:      POS transaction records
        activity_df: Patron activity events

    Returns:
        Master behavioral features DataFrame — one row per patron
    """
    logger.info("Building master behavioral feature set...")

    genre_df = extract_genre_features(lms_df)
    spending_df = extract_spending_features(pos_df)
    engagement_df = extract_engagement_features(activity_df)

    # Start with all patrons from LMS
    all_patrons = pd.DataFrame({"patron_id": lms_df["patron_id"].unique()})

    # Left join — keep all LMS patrons, fill missing POS/activity with 0
    features = all_patrons \
        .merge(genre_df, on="patron_id", how="left") \
        .merge(spending_df, on="patron_id", how="left") \
        .merge(engagement_df, on="patron_id", how="left")

    # Fill numeric nulls with 0, categorical with "unknown"
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    cat_cols = features.select_dtypes(include=["object"]).columns
    cat_cols = [c for c in cat_cols if c != "patron_id"]
    features[cat_cols] = features[cat_cols].fillna("unknown")

    logger.info(
        f"Built {len(features.columns) - 1} behavioral features "
        f"for {len(features)} patrons"
    )
    return features


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python features/behavioral_features.py
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingestion"))

    from lms_connector import load_lms_data
    from pos_connector import load_pos_data
    from kafka_consumer import load_activity_events

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=5000)
    pos_df = load_pos_data(source="synthetic", n_transactions=2000)
    activity_df = load_activity_events(source="synthetic", n_events=2000)

    print("Building behavioral features...")
    features_df = build_behavioral_features(lms_df, pos_df, activity_df)

    print("\n── Shape ──")
    print(features_df.shape)
    print("\n── Columns ──")
    print(features_df.columns.tolist())
    print("\n── Sample ──")
    print(features_df.head(3))