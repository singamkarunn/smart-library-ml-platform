"""
tests/test_ingestion.py
------------------------
Unit tests for the ingestion pipeline.

Tests verify that:
- Data generators produce expected shapes and types
- Schema validation catches bad data correctly
- Edge cases (empty data, nulls) are handled gracefully

Run: pytest tests/test_ingestion.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from lms_connector import (
    load_lms_data, generate_synthetic_lms_data,
    validate_schema, REQUIRED_COLUMNS
)
from pos_connector import (
    load_pos_data, generate_synthetic_pos_data,
    validate_schema as validate_pos_schema
)
from kafka_consumer import (
    load_activity_events, simulate_kafka_consumer
)


# ── LMS Connector Tests ───────────────────────────────────────────────────

class TestLMSConnector:

    def test_generates_correct_row_count(self):
        """Generator produces exactly n_transactions rows."""
        df = generate_synthetic_lms_data(n_transactions=100, seed=42)
        assert len(df) == 100

    def test_has_all_required_columns(self):
        """Generated data contains all required schema columns."""
        df = generate_synthetic_lms_data(n_transactions=100, seed=42)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"

    def test_no_null_patron_ids(self):
        """patron_id column contains no null values."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert df["patron_id"].isnull().sum() == 0

    def test_no_null_book_ids(self):
        """book_id column contains no null values."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert df["book_id"].isnull().sum() == 0

    def test_return_date_after_checkout_date(self):
        """Return date is always after or equal to checkout date."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert (df["return_date"] >= df["checkout_date"]).all()

    def test_loan_duration_positive(self):
        """Loan duration is always positive."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert (df["loan_duration_days"] > 0).all()

    def test_valid_loan_statuses(self):
        """Loan status only contains expected values."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        valid_statuses = {"returned", "overdue", "lost"}
        assert set(df["loan_status"].unique()).issubset(valid_statuses)

    def test_valid_genres(self):
        """All genres are from the expected list."""
        from lms_connector import GENRES
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert set(df["genre"].unique()).issubset(set(GENRES))

    def test_renewals_non_negative(self):
        """times_renewed is always >= 0."""
        df = generate_synthetic_lms_data(n_transactions=500, seed=42)
        assert (df["times_renewed"] >= 0).all()

    def test_checkout_date_is_datetime(self):
        """checkout_date column is datetime type."""
        df = generate_synthetic_lms_data(n_transactions=100, seed=42)
        assert pd.api.types.is_datetime64_any_dtype(df["checkout_date"])

    def test_schema_validation_passes_on_good_data(self):
        """Schema validator returns True for valid data."""
        df = generate_synthetic_lms_data(n_transactions=100, seed=42)
        assert validate_schema(df) is True

    def test_schema_validation_fails_on_missing_column(self):
        """Schema validator raises ValueError when column is missing."""
        df = generate_synthetic_lms_data(n_transactions=100, seed=42)
        df = df.drop(columns=["patron_id"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_schema(df)

    def test_reproducibility_with_same_seed(self):
        """Same seed produces identical DataFrames."""
        df1 = generate_synthetic_lms_data(n_transactions=100, seed=42)
        df2 = generate_synthetic_lms_data(n_transactions=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Different seeds produce different DataFrames."""
        df1 = generate_synthetic_lms_data(n_transactions=100, seed=42)
        df2 = generate_synthetic_lms_data(n_transactions=100, seed=99)
        assert not df1["patron_id"].equals(df2["patron_id"])

    def test_load_lms_data_synthetic(self):
        """load_lms_data works with synthetic source."""
        df = load_lms_data(source="synthetic", n_transactions=100)
        assert len(df) == 100
        assert isinstance(df, pd.DataFrame)

    def test_load_lms_data_invalid_source_raises(self):
        """load_lms_data raises ValueError for unknown source."""
        with pytest.raises(ValueError, match="Unknown source"):
            load_lms_data(source="invalid_source")

    def test_load_lms_data_csv_without_path_raises(self):
        """load_lms_data raises ValueError when csv source has no filepath."""
        with pytest.raises(ValueError, match="filepath required"):
            load_lms_data(source="csv")


# ── POS Connector Tests ───────────────────────────────────────────────────

class TestPOSConnector:

    def test_generates_correct_row_count(self):
        """Generator produces exactly n_transactions rows."""
        df = generate_synthetic_pos_data(n_transactions=100, seed=42)
        assert len(df) == 100

    def test_no_negative_amounts(self):
        """All transaction amounts are positive."""
        df = generate_synthetic_pos_data(n_transactions=500, seed=42)
        assert (df["amount"] > 0).all()

    def test_no_duplicate_transaction_ids(self):
        """Transaction IDs are unique."""
        df = generate_synthetic_pos_data(n_transactions=500, seed=42)
        assert df["transaction_id"].duplicated().sum() == 0

    def test_valid_transaction_types(self):
        """Only expected transaction types appear."""
        from pos_connector import TRANSACTION_TYPES
        df = generate_synthetic_pos_data(n_transactions=500, seed=42)
        assert set(df["transaction_type"].unique()).issubset(set(TRANSACTION_TYPES))

    def test_items_purchased_positive(self):
        """items_purchased is always >= 1."""
        df = generate_synthetic_pos_data(n_transactions=500, seed=42)
        assert (df["items_purchased"] >= 1).all()

    def test_pos_schema_validation_passes(self):
        """Schema validator passes on valid POS data."""
        df = generate_synthetic_pos_data(n_transactions=100, seed=42)
        assert validate_pos_schema(df) is True

    def test_load_pos_data_synthetic(self):
        """load_pos_data works with synthetic source."""
        df = load_pos_data(source="synthetic", n_transactions=100)
        assert len(df) == 100


# ── Kafka Consumer Tests ──────────────────────────────────────────────────

class TestKafkaConsumer:

    def test_generates_correct_event_count(self):
        """Consumer produces exactly n_events events."""
        df = simulate_kafka_consumer(n_events=200, batch_size=50, seed=42)
        assert len(df) == 200

    def test_no_duplicate_event_ids(self):
        """Event IDs are unique across all batches."""
        df = simulate_kafka_consumer(n_events=500, batch_size=100, seed=42)
        assert df["event_id"].duplicated().sum() == 0

    def test_valid_event_types(self):
        """Only expected event types appear."""
        from kafka_consumer import EVENT_TYPES
        df = simulate_kafka_consumer(n_events=500, batch_size=100, seed=42)
        assert set(df["event_type"].unique()).issubset(set(EVENT_TYPES))

    def test_no_null_patron_ids(self):
        """patron_id column has no nulls."""
        df = simulate_kafka_consumer(n_events=200, seed=42)
        assert df["patron_id"].isnull().sum() == 0

    def test_batch_processing_produces_same_result(self):
        """Different batch sizes produce same total event count."""
        df1 = simulate_kafka_consumer(n_events=200, batch_size=50, seed=42)
        df2 = simulate_kafka_consumer(n_events=200, batch_size=200, seed=42)
        assert len(df1) == len(df2)

    def test_session_events_have_no_book_id(self):
        """Session start/end events correctly have null book_id."""
        df = simulate_kafka_consumer(n_events=500, seed=42)
        session_events = df[df["event_type"].isin(["session_start", "session_end"])]
        assert session_events["book_id"].isnull().all()

    def test_load_activity_events_synthetic(self):
        """load_activity_events works with synthetic source."""
        df = load_activity_events(source="synthetic", n_events=100)
        assert len(df) == 100