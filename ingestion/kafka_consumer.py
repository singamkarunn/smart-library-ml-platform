"""
ingestion/kafka_consumer.py
---------------------------
Simulates real-time user activity ingestion via Kafka.

In production: connects to a real Kafka broker and consumes
events as patrons interact with the library system in real time —
book searches, page views, wishlist additions, session activity.

In development: generates synthetic activity events and simulates
the Kafka consumer loop so you can test the full pipeline locally
without needing a running Kafka cluster.

Output: Streams activity events into a pandas DataFrame, one batch
        at a time, ready for real-time feature updates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
EVENT_TYPES = [
    "book_search",       # patron searched for a book
    "book_view",         # patron viewed a book detail page
    "wishlist_add",      # patron added book to wishlist
    "wishlist_remove",   # patron removed book from wishlist
    "reservation_made",  # patron reserved a book
    "reservation_cancel",# patron cancelled a reservation
    "review_submitted",  # patron submitted a book review
    "session_start",     # patron logged into the system
    "session_end"        # patron logged out
]

# Weight reflects real-world frequency:
# searches and views dominate; reviews are rare
EVENT_WEIGHTS = [0.30, 0.25, 0.12, 0.04, 0.10, 0.03, 0.02, 0.07, 0.07]

REQUIRED_COLUMNS = [
    "event_id", "patron_id", "event_type",
    "event_timestamp", "book_id", "session_id",
    "device_type", "event_metadata"
]


# ── Event Generator ───────────────────────────────────────────────────────
def generate_activity_event(patron_ids: list,
                             book_ids: list,
                             base_timestamp: datetime,
                             event_counter: int) -> dict:
    """
    Generates a single synthetic patron activity event.

    Each event mimics what a real Kafka message would contain —
    a patron doing something in the library system at a specific moment.

    Args:
        patron_ids:      Pool of patron IDs to sample from
        book_ids:        Pool of book IDs to sample from
        base_timestamp:  Starting point for event timestamp
        event_counter:   Used to generate unique event IDs

    Returns:
        dict representing one activity event (one Kafka message)
    """
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
    patron_id = random.choice(patron_ids)

    # Not all events are tied to a specific book
    # Session start/end and some searches have no book_id
    book_id = random.choice(book_ids) if event_type not in [
        "session_start", "session_end"
    ] else None

    # Simulate realistic timestamp offsets (events arrive seconds apart)
    offset_seconds = random.randint(1, 30)
    event_timestamp = base_timestamp + timedelta(seconds=offset_seconds * event_counter)

    # Metadata varies by event type — search has a query, review has a rating
    metadata = {}
    if event_type == "book_search":
        metadata["query_length"] = random.randint(2, 8)
        metadata["results_shown"] = random.randint(0, 20)
    elif event_type == "review_submitted":
        metadata["rating"] = random.randint(1, 5)
        metadata["review_length_words"] = random.randint(10, 300)
    elif event_type in ["session_start", "session_end"]:
        metadata["session_duration_minutes"] = random.randint(1, 120)

    return {
        "event_id": f"E{str(event_counter).zfill(8)}",
        "patron_id": patron_id,
        "event_type": event_type,
        "event_timestamp": event_timestamp,
        "book_id": book_id,
        "session_id": f"S{random.randint(10000, 99999)}",
        "device_type": random.choices(
            ["desktop", "mobile", "tablet"],
            weights=[0.55, 0.35, 0.10]
        )[0],
        "event_metadata": json.dumps(metadata)
    }


# ── Simulated Kafka Consumer ──────────────────────────────────────────────
def simulate_kafka_consumer(n_patrons: int = 500,
                              n_books: int = 1000,
                              n_events: int = 2000,
                              batch_size: int = 100,
                              seed: int = 42) -> pd.DataFrame:
    """
    Simulates consuming activity events from a Kafka topic in batches.

    Why batch processing?
    In production, Kafka consumers read messages in configurable batches.
    This function mimics that behavior — processing events in groups of
    batch_size, which is how the real pipeline will work.

    Args:
        n_patrons:   Number of unique patrons generating events
        n_books:     Number of books events can reference
        n_events:    Total events to generate
        batch_size:  How many events to process per batch
        seed:        Random seed for reproducibility

    Returns:
        pd.DataFrame of all consumed events
    """
    random.seed(seed)
    np.random.seed(seed)

    patron_ids = [f"P{str(i).zfill(5)}" for i in range(n_patrons)]
    book_ids = [f"B{str(i).zfill(5)}" for i in range(n_books)]
    base_timestamp = datetime(2024, 1, 1, 8, 0, 0)

    logger.info(f"Starting Kafka consumer simulation — {n_events} events in batches of {batch_size}")

    all_events = []
    n_batches = (n_events + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = []

        for i in range(batch_start, batch_end):
            event = generate_activity_event(
                patron_ids, book_ids, base_timestamp, i
            )
            batch_events.append(event)

        all_events.extend(batch_events)
        logger.info(f"Batch {batch_num + 1}/{n_batches} consumed — {len(batch_events)} events")

    df = pd.DataFrame(all_events)
    logger.info(
        f"Consumer finished — {len(df)} total events from "
        f"{df['patron_id'].nunique()} unique patrons"
    )
    return df


# ── Schema Validator ──────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> bool:
    """Validates activity event DataFrame has required columns."""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing: {missing_cols}")

    if df["event_id"].duplicated().any():
        raise ValueError("event_id contains duplicates")

    if df["patron_id"].isnull().any():
        raise ValueError("patron_id cannot contain nulls")

    logger.info("Activity event schema validation passed")
    return True


# ── Main Loader ───────────────────────────────────────────────────────────
def load_activity_events(source: str = "synthetic",
                          filepath: str = None,
                          **kwargs) -> pd.DataFrame:
    """
    Main entry point for loading patron activity events.

    Args:
        source:   "synthetic" or "csv"
        filepath: path to CSV (required if source="csv")
        **kwargs: passed to simulate_kafka_consumer()

    Returns:
        Validated pd.DataFrame of activity events
    """
    logger.info(f"Loading activity events from source: {source}")

    if source == "synthetic":
        df = simulate_kafka_consumer(**kwargs)

    elif source == "csv":
        if filepath is None:
            raise ValueError("filepath required when source='csv'")
        df = pd.read_csv(filepath, parse_dates=["event_timestamp"])
        logger.info(f"Loaded {len(df)} events from {filepath}")

    else:
        raise ValueError(f"Unknown source: '{source}'. Use 'synthetic' or 'csv'")

    validate_schema(df)
    return df


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python ingestion/kafka_consumer.py
if __name__ == "__main__":
    df = load_activity_events(source="synthetic", n_events=500, batch_size=100)
    print("\n── Sample Events ──")
    print(df.head())
    print("\n── Event Type Distribution ──")
    print(df["event_type"].value_counts())
    print("\n── Device Types ──")
    print(df["device_type"].value_counts())
    print(f"\nEvents with book_id: {df['book_id'].notna().sum()}")
    print(f"Events without book_id: {df['book_id'].isna().sum()}")