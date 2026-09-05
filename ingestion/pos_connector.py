"""
ingestion/pos_connector.py
--------------------------
Loads and validates Point-of-Sale (POS) transaction records.

POS data captures what patrons purchase at the library — cafe items,
merchandise, event tickets, printing fees. Combined with borrowing data
it reveals patron engagement beyond just book checkouts.

Output: A clean, validated pandas DataFrame ready for feature engineering.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
TRANSACTION_TYPES = ["cafe", "merchandise", "printing", "event_ticket", "donation"]

PAYMENT_METHODS = ["card", "cash", "mobile"]

REQUIRED_COLUMNS = [
    "transaction_id", "patron_id", "transaction_date",
    "transaction_type", "amount", "payment_method", "items_purchased"
]


# ── Synthetic Generator ───────────────────────────────────────────────────
def generate_synthetic_pos_data(n_patrons: int = 500,
                                 n_transactions: int = 5000,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic POS transaction records.

    Key behavioral insight baked in:
    - Frequent borrowers tend to spend more at the library cafe
    - Premium members buy more merchandise
    - Event tickets are rare but high-value transactions

    Args:
        n_patrons:      Number of unique patrons (should match LMS)
        n_transactions: Total POS transactions to generate
        seed:           Random seed for reproducibility

    Returns:
        pd.DataFrame with columns matching REQUIRED_COLUMNS
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating {n_transactions} synthetic POS transactions...")

    patron_ids = [f"P{str(i).zfill(5)}" for i in range(n_patrons)]

    # Amount ranges by transaction type
    # Event tickets cost more than coffee — model that explicitly
    amount_ranges = {
        "cafe":          (2.50, 12.00),
        "merchandise":   (5.00, 45.00),
        "printing":      (0.10, 8.00),
        "event_ticket":  (10.00, 75.00),
        "donation":      (1.00, 100.00)
    }

    # Transaction type weights — cafe is most common
    type_weights = [0.45, 0.20, 0.25, 0.05, 0.05]

    records = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    for i in range(n_transactions):
        patron_id = random.choice(patron_ids)
        transaction_type = random.choices(
            TRANSACTION_TYPES,
            weights=type_weights
        )[0]

        amount_min, amount_max = amount_ranges[transaction_type]
        amount = round(random.uniform(amount_min, amount_max), 2)

        transaction_date = start_date + timedelta(
            days=random.randint(0, date_range),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59)
        )

        # Items purchased — 1 for most, occasionally 2-3
        items_purchased = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]

        records.append({
            "transaction_id": f"T{str(i).zfill(6)}",
            "patron_id": patron_id,
            "transaction_date": transaction_date,
            "transaction_type": transaction_type,
            "amount": amount,
            "payment_method": random.choices(
                PAYMENT_METHODS,
                weights=[0.60, 0.25, 0.15]
            )[0],
            "items_purchased": items_purchased
        })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} POS transactions for "
        f"{df['patron_id'].nunique()} unique patrons — "
        f"total revenue: ${df['amount'].sum():,.2f}"
    )
    return df


# ── Schema Validator ──────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> bool:
    """Validates POS DataFrame has required columns and clean data."""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing: {missing_cols}")

    if df["amount"].lt(0).any():
        raise ValueError("amount contains negative values")

    if df["transaction_id"].duplicated().any():
        raise ValueError("transaction_id contains duplicates")

    if df["patron_id"].isnull().any():
        raise ValueError("patron_id cannot contain nulls")

    logger.info("POS schema validation passed")
    return True


# ── Main Loader ───────────────────────────────────────────────────────────
def load_pos_data(source: str = "synthetic",
                  filepath: str = None,
                  **kwargs) -> pd.DataFrame:
    """
    Main entry point for loading POS transaction data.

    Args:
        source:   "synthetic" or "csv"
        filepath: path to CSV (required if source="csv")
        **kwargs: passed to generate_synthetic_pos_data()

    Returns:
        Validated pd.DataFrame ready for feature engineering
    """
    logger.info(f"Loading POS data from source: {source}")

    if source == "synthetic":
        df = generate_synthetic_pos_data(**kwargs)

    elif source == "csv":
        if filepath is None:
            raise ValueError("filepath required when source='csv'")
        df = pd.read_csv(filepath, parse_dates=["transaction_date"])
        logger.info(f"Loaded {len(df)} POS records from {filepath}")

    else:
        raise ValueError(f"Unknown source: '{source}'. Use 'synthetic' or 'csv'")

    validate_schema(df)
    return df


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python ingestion/pos_connector.py
if __name__ == "__main__":
    df = load_pos_data(source="synthetic", n_transactions=1000)
    print("\n── Sample Data ──")
    print(df.head())
    print("\n── Transaction Types ──")
    print(df["transaction_type"].value_counts())
    print("\n── Revenue by Type ──")
    print(df.groupby("transaction_type")["amount"].sum().round(2).sort_values(ascending=False))
    print(f"\nTotal Revenue: ${df['amount'].sum():,.2f}")