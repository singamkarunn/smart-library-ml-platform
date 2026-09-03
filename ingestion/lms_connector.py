"""
ingestion/lms_connector.py
--------------------------
Loads and validates Library Management System (LMS) borrowing records.

In production: connects to a real LMS database via SQLAlchemy.
In development: generates realistic synthetic data using Faker so you
can build and test the full pipeline without needing real library data.

Output: A clean, validated pandas DataFrame with standardized column names
        ready to be passed into the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import logging

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
GENRES = [
    "Fiction", "Non-Fiction", "Science Fiction", "Mystery",
    "Biography", "History", "Technology", "Children", "Romance", "Philosophy"
]

LOAN_STATUSES = ["returned", "overdue", "lost"]

# Every column the downstream pipeline expects.
# If data doesn't match this, we catch it early rather than letting
# it silently corrupt model training.
REQUIRED_COLUMNS = [
    "patron_id", "book_id", "book_title", "genre",
    "checkout_date", "return_date", "loan_status",
    "loan_duration_days", "times_renewed",
    "patron_age_group", "patron_membership_type"
]


# ── Synthetic Data Generator ──────────────────────────────────────────────
def generate_synthetic_lms_data(n_patrons: int = 500,
                                 n_books: int = 1000,
                                 n_transactions: int = 10000,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic LMS borrowing records.

    Why synthetic data?
    - Lets you build and test the full pipeline without real library data
    - Faker produces realistic names, dates, and patterns
    - Controlled seed means reproducible results for testing

    Args:
        n_patrons:      Number of unique library patrons to simulate
        n_books:        Number of unique books in the catalog
        n_transactions: Total borrowing transactions to generate
        seed:           Random seed for reproducibility

    Returns:
        pd.DataFrame with columns matching REQUIRED_COLUMNS
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating {n_transactions} synthetic LMS transactions...")

    # Generate patron pool
    # Real patron behavior: some patrons borrow frequently, most rarely.
    # Zipf distribution models this — a few heavy users dominate.
    patron_ids = [f"P{str(i).zfill(5)}" for i in range(n_patrons)]
    patron_age_groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
    patron_membership_types = ["basic", "standard", "premium"]

    patron_metadata = {
        pid: {
            "age_group": random.choice(patron_age_groups),
            "membership_type": random.choices(
                patron_membership_types,
                weights=[0.5, 0.35, 0.15]
            )[0]
        }
        for pid in patron_ids
    }

    # Generate book catalog
    book_ids = [f"B{str(i).zfill(5)}" for i in range(n_books)]
    book_metadata = {
        bid: {
            "title": fake.catch_phrase(),
            "genre": random.choice(GENRES)
        }
        for bid in book_ids
    }

    # Heavy-tailed patron selection: some patrons borrow a lot
    patron_weights = np.random.zipf(1.5, n_patrons)
    patron_weights = patron_weights / patron_weights.sum()

    records = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    for _ in range(n_transactions):
        patron_id = np.random.choice(patron_ids, p=patron_weights)
        book_id = random.choice(book_ids)

        checkout_date = start_date + timedelta(days=random.randint(0, date_range))
        loan_duration = max(1, int(np.random.normal(14, 7)))
        return_date = checkout_date + timedelta(days=loan_duration)

        loan_status = random.choices(
            LOAN_STATUSES,
            weights=[0.85, 0.12, 0.03]
        )[0]

        membership = patron_metadata[patron_id]["membership_type"]
        max_renewals = {"basic": 1, "standard": 2, "premium": 4}[membership]
        times_renewed = random.randint(0, max_renewals)

        records.append({
            "patron_id": patron_id,
            "book_id": book_id,
            "book_title": book_metadata[book_id]["title"],
            "genre": book_metadata[book_id]["genre"],
            "checkout_date": checkout_date,
            "return_date": return_date,
            "loan_status": loan_status,
            "loan_duration_days": loan_duration,
            "times_renewed": times_renewed,
            "patron_age_group": patron_metadata[patron_id]["age_group"],
            "patron_membership_type": membership
        })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} transactions for "
        f"{df['patron_id'].nunique()} patrons and "
        f"{df['book_id'].nunique()} books"
    )
    return df


# ── Schema Validator ──────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validates the DataFrame has all required columns and correct types.
    Schema errors caught at ingestion save hours of debugging downstream.
    """
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing columns: {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["checkout_date"]):
        raise ValueError("checkout_date must be datetime type")

    if df["patron_id"].isnull().any():
        raise ValueError("patron_id cannot contain null values")

    if df["book_id"].isnull().any():
        raise ValueError("book_id cannot contain null values")

    logger.info("Schema validation passed")
    return True


# ── Main Loader ───────────────────────────────────────────────────────────
def load_lms_data(source: str = "synthetic",
                  filepath: str = None,
                  **kwargs) -> pd.DataFrame:
    """
    Main entry point for loading LMS data.

    Args:
        source:   "synthetic" or "csv"
        filepath: path to CSV file (required if source="csv")
        **kwargs: passed to generate_synthetic_lms_data()

    Returns:
        Validated pd.DataFrame ready for feature engineering
    """
    logger.info(f"Loading LMS data from source: {source}")

    if source == "synthetic":
        df = generate_synthetic_lms_data(**kwargs)

    elif source == "csv":
        if filepath is None:
            raise ValueError("filepath required when source='csv'")
        df = pd.read_csv(filepath, parse_dates=["checkout_date", "return_date"])
        logger.info(f"Loaded {len(df)} records from {filepath}")

    else:
        raise ValueError(f"Unknown source: '{source}'. Use 'synthetic' or 'csv'")

    validate_schema(df)
    return df


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python ingestion/lms_connector.py
if __name__ == "__main__":
    df = load_lms_data(source="synthetic", n_transactions=1000)
    print("\n── Sample Data ──")
    print(df.head())
    print("\n── Data Types ──")
    print(df.dtypes)
    print("\n── Summary ──")
    print(f"Patrons: {df['patron_id'].nunique()}")
    print(f"Books:   {df['book_id'].nunique()}")
    print(f"Genres:  {df['genre'].value_counts().to_dict()}")