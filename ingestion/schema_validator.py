"""
ingestion/schema_validator.py
------------------------------
Central schema validation hub for all three data sources.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_lms(df: pd.DataFrame) -> dict:
    results = {}
    null_patrons = df["patron_id"].isnull().sum()
    results["no_null_patron_ids"] = {"passed": null_patrons == 0, "detail": f"{null_patrons} null patron_ids found"}
    invalid_dates = (df["return_date"] < df["checkout_date"]).sum()
    results["return_after_checkout"] = {"passed": invalid_dates == 0, "detail": f"{invalid_dates} records where return_date < checkout_date"}
    invalid_duration = (df["loan_duration_days"] <= 0).sum()
    results["positive_loan_duration"] = {"passed": invalid_duration == 0, "detail": f"{invalid_duration} records with non-positive loan duration"}
    valid_statuses = {"returned", "overdue", "lost"}
    invalid_statuses = (~df["loan_status"].isin(valid_statuses)).sum()
    results["valid_loan_status"] = {"passed": invalid_statuses == 0, "detail": f"{invalid_statuses} records with invalid loan_status"}
    invalid_renewals = (df["times_renewed"] < 0).sum()
    results["non_negative_renewals"] = {"passed": invalid_renewals == 0, "detail": f"{invalid_renewals} records with negative renewals"}
    return results


def validate_pos(df: pd.DataFrame) -> dict:
    results = {}
    dupes = df["transaction_id"].duplicated().sum()
    results["no_duplicate_transactions"] = {"passed": dupes == 0, "detail": f"{dupes} duplicate transaction_ids"}
    negative_amounts = (df["amount"] < 0).sum()
    results["no_negative_amounts"] = {"passed": negative_amounts == 0, "detail": f"{negative_amounts} records with negative amount"}
    valid_types = {"cafe", "merchandise", "printing", "event_ticket", "donation"}
    invalid_types = (~df["transaction_type"].isin(valid_types)).sum()
    results["valid_transaction_types"] = {"passed": invalid_types == 0, "detail": f"{invalid_types} records with invalid transaction_type"}
    invalid_items = (df["items_purchased"] <= 0).sum()
    results["positive_items_purchased"] = {"passed": invalid_items == 0, "detail": f"{invalid_items} records with non-positive items_purchased"}
    return results


def validate_activity(df: pd.DataFrame) -> dict:
    results = {}
    dupes = df["event_id"].duplicated().sum()
    results["no_duplicate_events"] = {"passed": dupes == 0, "detail": f"{dupes} duplicate event_ids"}
    valid_events = {"book_search", "book_view", "wishlist_add", "wishlist_remove",
                    "reservation_made", "reservation_cancel", "review_submitted",
                    "session_start", "session_end"}
    invalid_events = (~df["event_type"].isin(valid_events)).sum()
    results["valid_event_types"] = {"passed": invalid_events == 0, "detail": f"{invalid_events} records with invalid event_type"}
    null_patrons = df["patron_id"].isnull().sum()
    results["no_null_patron_ids"] = {"passed": null_patrons == 0, "detail": f"{null_patrons} null patron_ids"}
    return results


def validate_cross_source(lms_df: pd.DataFrame,
                           pos_df: pd.DataFrame,
                           activity_df: pd.DataFrame) -> dict:
    results = {}
    lms_patrons = set(lms_df["patron_id"].unique())
    pos_patrons = set(pos_df["patron_id"].unique())
    activity_patrons = set(activity_df["patron_id"].unique())

    pos_only = pos_patrons - lms_patrons
    results["pos_patrons_in_lms"] = {"passed": len(pos_only) == 0, "detail": f"{len(pos_only)} patrons in POS not found in LMS"}

    activity_only = activity_patrons - lms_patrons
    results["activity_patrons_in_lms"] = {"passed": len(activity_only) == 0, "detail": f"{len(activity_only)} patrons in activity not found in LMS"}

    lms_min = lms_df["checkout_date"].min()
    lms_max = lms_df["checkout_date"].max()
    pos_min = pos_df["transaction_date"].min()
    pos_max = pos_df["transaction_date"].max()
    overlap = (pos_min <= lms_max) and (pos_max >= lms_min)
    results["date_range_overlap"] = {"passed": overlap, "detail": f"LMS: {lms_min.date()} to {lms_max.date()} | POS: {pos_min.date()} to {pos_max.date()}"}
    return results


def print_validation_report(reports: dict) -> bool:
    print("\n" + "="*60)
    print("  DATA VALIDATION REPORT")
    print("="*60)
    all_passed = True
    for source, checks in reports.items():
        print(f"\n── {source.upper()} ──")
        for check_name, result in checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status}  {check_name}")
            if not result["passed"]:
                print(f"         → {result['detail']}")
                all_passed = False
    print("\n" + "="*60)
    overall = "✅ ALL CHECKS PASSED" if all_passed else "❌ VALIDATION FAILED"
    print(f"  {overall}")
    print("="*60 + "\n")
    return all_passed


def run_full_validation(lms_df: pd.DataFrame,
                        pos_df: pd.DataFrame,
                        activity_df: pd.DataFrame) -> bool:
    logger.info("Running full data validation across all sources...")
    reports = {
        "lms": validate_lms(lms_df),
        "pos": validate_pos(pos_df),
        "activity": validate_activity(activity_df),
        "cross_source": validate_cross_source(lms_df, pos_df, activity_df)
    }
    passed = print_validation_report(reports)
    if passed:
        logger.info("All checks passed — data ready for feature engineering")
    else:
        logger.warning("Validation failed — fix issues before proceeding")
    return passed


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lms_connector import load_lms_data
    from pos_connector import load_pos_data
    from kafka_consumer import load_activity_events

    print("Loading data...")
    lms_df = load_lms_data(source="synthetic", n_transactions=1000)
    pos_df = load_pos_data(source="synthetic", n_transactions=500)
    activity_df = load_activity_events(source="synthetic", n_events=500)

    print("Running validation...")
    run_full_validation(lms_df, pos_df, activity_df)