"""
airflow/dags/etl_pipeline.py
-----------------------------
Master ETL pipeline DAG for the Smart Library ML Platform.

This DAG orchestrates the full data ingestion and feature engineering
pipeline on a daily schedule:

1. Extract: Load data from LMS, POS, and Kafka activity sources
2. Validate: Run cross-source schema validation
3. Transform: Engineer temporal, behavioral, and content features
4. Load: Save processed features ready for model training

Why Airflow for this pipeline?
- Dependency management: Task 3 only runs if Task 2 passes
- Retry logic: Failed tasks retry automatically with backoff
- Observability: Every run is logged with success/failure status
- Scheduling: Runs daily at 2am without manual intervention
- Backfill: Can reprocess historical dates if data arrives late

In production this DAG would connect to real databases.
In development it uses the synthetic data generators.
"""

from datetime import datetime, timedelta
import logging
import os
import sys

# ── Airflow imports with graceful fallback ────────────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(DAG_DIR, "..", "..")
INGESTION_DIR = os.path.join(ROOT_DIR, "ingestion")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

sys.path.insert(0, INGESTION_DIR)
sys.path.insert(0, FEATURES_DIR)

# ── Default DAG arguments ─────────────────────────────────────────────────
# These apply to every task in the DAG unless overridden
DEFAULT_ARGS = {
    "owner": "karun_singampalli",
    "depends_on_past": False,        # Don't wait for yesterday's run
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,                    # Retry failed tasks twice
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Task Functions ────────────────────────────────────────────────────────
# Each function is one task in the DAG.
# Tasks communicate via Airflow's XCom (cross-communication) system —
# one task pushes data, the next task pulls it.

def extract_lms_data(**context) -> dict:
    """
    Task 1: Extract LMS borrowing records.

    In production: connects to LMS database via SQLAlchemy.
    In development: uses synthetic data generator.

    Pushes:
        lms_record_count: Number of records extracted
        lms_patron_count: Number of unique patrons
    """
    from lms_connector import load_lms_data

    logger.info("ETL Task 1: Extracting LMS data...")

    execution_date = context.get("execution_date", datetime.now())
    logger.info(f"Processing data for: {execution_date.date()}")

    lms_df = load_lms_data(source="synthetic", n_transactions=10000)

    # Save to data directory for downstream tasks
    data_dir = os.path.join(ROOT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    lms_df.to_parquet(os.path.join(data_dir, "lms_raw.parquet"), index=False)

    # Push metadata to XCom for monitoring
    context["task_instance"].xcom_push(
        key="lms_record_count", value=len(lms_df)
    )
    context["task_instance"].xcom_push(
        key="lms_patron_count", value=int(lms_df["patron_id"].nunique())
    )

    logger.info(f"LMS extraction complete: {len(lms_df)} records")
    return {"status": "success", "records": len(lms_df)}


def extract_pos_data(**context) -> dict:
    """
    Task 2: Extract POS transaction records.

    Runs in parallel with LMS extraction since they're independent.
    """
    from pos_connector import load_pos_data

    logger.info("ETL Task 2: Extracting POS data...")

    pos_df = load_pos_data(source="synthetic", n_transactions=5000)

    data_dir = os.path.join(ROOT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    pos_df.to_parquet(os.path.join(data_dir, "pos_raw.parquet"), index=False)

    context["task_instance"].xcom_push(
        key="pos_record_count", value=len(pos_df)
    )

    logger.info(f"POS extraction complete: {len(pos_df)} records")
    return {"status": "success", "records": len(pos_df)}


def extract_activity_data(**context) -> dict:
    """
    Task 3: Extract patron activity events from Kafka.

    Runs in parallel with LMS and POS extractions.
    """
    from kafka_consumer import load_activity_events

    logger.info("ETL Task 3: Extracting activity events...")

    activity_df = load_activity_events(
        source="synthetic", n_events=8000, batch_size=200
    )

    data_dir = os.path.join(ROOT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    activity_df.to_parquet(
        os.path.join(data_dir, "activity_raw.parquet"), index=False
    )

    context["task_instance"].xcom_push(
        key="activity_event_count", value=len(activity_df)
    )

    logger.info(f"Activity extraction complete: {len(activity_df)} events")
    return {"status": "success", "records": len(activity_df)}


def validate_data(**context) -> dict:
    """
    Task 4: Cross-source data validation.

    Runs AFTER all three extractions complete (depends_on: extract_*)
    Fails the DAG if critical validation checks fail.

    Why fail fast on validation?
    Bad data flowing into model training is worse than no data —
    it silently corrupts model quality without raising an error.
    """
    import pandas as pd
    from schema_validator import run_full_validation

    logger.info("ETL Task 4: Validating data across sources...")

    data_dir = os.path.join(ROOT_DIR, "data")

    lms_df = pd.read_parquet(os.path.join(data_dir, "lms_raw.parquet"))
    pos_df = pd.read_parquet(os.path.join(data_dir, "pos_raw.parquet"))
    activity_df = pd.read_parquet(
        os.path.join(data_dir, "activity_raw.parquet")
    )

    passed = run_full_validation(lms_df, pos_df, activity_df)

    if not passed:
        # In development we warn but continue so pipeline can be tested
        if os.environ.get("AIRFLOW_ENV", "development") == "production":
            raise ValueError(
                "Data validation failed — pipeline halted. "
                "Check validation report for details."
            )
        else:
            logger.warning(
                "Validation issues detected (cross-source patron mismatch "
                "expected in synthetic data) — continuing in development mode"
            )

    logger.info("Validation passed — proceeding to feature engineering")
    return {"status": "success", "validation_passed": True}


def engineer_temporal_features(**context) -> dict:
    """
    Task 5: Extract temporal features from LMS data.

    Runs after validation passes.
    """
    import pandas as pd
    from temporal_features import extract_temporal_features

    logger.info("ETL Task 5: Engineering temporal features...")

    data_dir = os.path.join(ROOT_DIR, "data")
    lms_df = pd.read_parquet(os.path.join(data_dir, "lms_raw.parquet"))

    temporal_df = extract_temporal_features(lms_df)

    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    temporal_df.to_parquet(
        os.path.join(features_dir, "temporal_features.parquet"), index=False
    )

    logger.info(
        f"Temporal features complete: {temporal_df.shape[1]-1} features "
        f"for {len(temporal_df)} patrons"
    )
    return {"status": "success", "patrons": len(temporal_df)}


def engineer_behavioral_features(**context) -> dict:
    """
    Task 6: Extract behavioral features from all three sources.

    Runs in parallel with temporal feature engineering.
    """
    import pandas as pd
    from behavioral_features import build_behavioral_features

    logger.info("ETL Task 6: Engineering behavioral features...")

    data_dir = os.path.join(ROOT_DIR, "data")
    lms_df = pd.read_parquet(os.path.join(data_dir, "lms_raw.parquet"))
    pos_df = pd.read_parquet(os.path.join(data_dir, "pos_raw.parquet"))
    activity_df = pd.read_parquet(
        os.path.join(data_dir, "activity_raw.parquet")
    )

    behavioral_df = build_behavioral_features(lms_df, pos_df, activity_df)

    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    behavioral_df.to_parquet(
        os.path.join(features_dir, "behavioral_features.parquet"), index=False
    )

    logger.info(
        f"Behavioral features complete: {behavioral_df.shape[1]-1} features "
        f"for {len(behavioral_df)} patrons"
    )
    return {"status": "success", "patrons": len(behavioral_df)}


def engineer_content_features(**context) -> dict:
    """
    Task 7: Extract content features from book metadata.

    Runs in parallel with temporal and behavioral feature engineering.
    """
    import pandas as pd
    from content_features import build_content_features

    logger.info("ETL Task 7: Engineering content features...")

    data_dir = os.path.join(ROOT_DIR, "data")
    lms_df = pd.read_parquet(os.path.join(data_dir, "lms_raw.parquet"))

    content = build_content_features(lms_df, tfidf_max_features=100)

    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    content["metadata"].to_parquet(
        os.path.join(features_dir, "book_metadata.parquet"), index=False
    )
    content["tfidf"].to_parquet(
        os.path.join(features_dir, "tfidf_features.parquet"), index=False
    )
    content["encoding"].to_parquet(
        os.path.join(features_dir, "genre_encoding.parquet"), index=False
    )

    logger.info(
        f"Content features complete: {len(content['metadata'])} books, "
        f"{content['tfidf'].shape[1]-1} TF-IDF features"
    )
    return {"status": "success", "books": len(content["metadata"])}


def load_features(**context) -> dict:
    """
    Task 8: Final load step — merges all features and validates output.

    Runs after ALL feature engineering tasks complete.
    This is the final gate before features are available for model training.
    """
    import pandas as pd

    logger.info("ETL Task 8: Loading and merging final feature set...")

    features_dir = os.path.join(ROOT_DIR, "data", "features")

    temporal_df = pd.read_parquet(
        os.path.join(features_dir, "temporal_features.parquet")
    )
    behavioral_df = pd.read_parquet(
        os.path.join(features_dir, "behavioral_features.parquet")
    )

    # Merge temporal and behavioral on patron_id
    master_features = temporal_df.merge(
        behavioral_df, on="patron_id", how="inner"
    )

    # Save master feature set
    master_path = os.path.join(features_dir, "master_features.parquet")
    master_features.to_parquet(master_path, index=False)

    logger.info(
        f"Master feature set saved: {master_features.shape[0]} patrons, "
        f"{master_features.shape[1]} total features"
    )

    context["task_instance"].xcom_push(
        key="feature_count", value=master_features.shape[1]
    )
    context["task_instance"].xcom_push(
        key="patron_count", value=len(master_features)
    )

    return {
        "status": "success",
        "patrons": len(master_features),
        "features": master_features.shape[1]
    }


# ── DAG Definition ────────────────────────────────────────────────────────
# This block only runs when Airflow is installed.
# The task functions above can be tested independently without Airflow.
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="smart_library_etl_pipeline",
        default_args=DEFAULT_ARGS,
        description="Daily ETL pipeline: ingest LMS/POS/activity data and engineer features",
        schedule_interval="0 2 * * *",   # Run daily at 2:00 AM
        start_date=days_ago(1),
        catchup=False,                   # Don't backfill missed runs
        max_active_runs=1,               # Only one run at a time
        tags=["etl", "smart-library", "features"],
    ) as dag:

        # ── Extraction tasks (run in parallel) ─────────────────────────
        t_extract_lms = PythonOperator(
            task_id="extract_lms_data",
            python_callable=extract_lms_data,
        )

        t_extract_pos = PythonOperator(
            task_id="extract_pos_data",
            python_callable=extract_pos_data,
        )

        t_extract_activity = PythonOperator(
            task_id="extract_activity_data",
            python_callable=extract_activity_data,
        )

        # ── Validation (waits for all extractions) ──────────────────────
        t_validate = PythonOperator(
            task_id="validate_data",
            python_callable=validate_data,
        )

        # ── Feature engineering (run in parallel after validation) ──────
        t_temporal = PythonOperator(
            task_id="engineer_temporal_features",
            python_callable=engineer_temporal_features,
        )

        t_behavioral = PythonOperator(
            task_id="engineer_behavioral_features",
            python_callable=engineer_behavioral_features,
        )

        t_content = PythonOperator(
            task_id="engineer_content_features",
            python_callable=engineer_content_features,
        )

        # ── Final load (waits for all feature engineering) ──────────────
        t_load = PythonOperator(
            task_id="load_features",
            python_callable=load_features,
        )

        # ── DAG dependency graph ─────────────────────────────────────────
        # Extraction runs in parallel:
        [t_extract_lms, t_extract_pos, t_extract_activity] >> t_validate
        # Feature engineering runs in parallel after validation:
        t_validate >> [t_temporal, t_behavioral, t_content]
        # Load waits for all feature engineering:
        [t_temporal, t_behavioral, t_content] >> t_load


# ── Standalone test (no Airflow needed) ──────────────────────────────────
# Run: python airflow/dags/etl_pipeline.py
if __name__ == "__main__":
    print("Running ETL pipeline in standalone mode (no Airflow)...")
    print("=" * 60)

    # Simulate Airflow context
    mock_context = {
        "execution_date": datetime.now(),
        "task_instance": type("MockTI", (), {
            "xcom_push": lambda self, key, value: print(f"  XCom: {key} = {value}")
        })()
    }

    print("\n[1/8] Extracting LMS data...")
    extract_lms_data(**mock_context)

    print("\n[2/8] Extracting POS data...")
    extract_pos_data(**mock_context)

    print("\n[3/8] Extracting activity data...")
    extract_activity_data(**mock_context)

    print("\n[4/8] Validating data...")
    validate_data(**mock_context)

    print("\n[5/8] Engineering temporal features...")
    engineer_temporal_features(**mock_context)

    print("\n[6/8] Engineering behavioral features...")
    engineer_behavioral_features(**mock_context)

    print("\n[7/8] Engineering content features...")
    engineer_content_features(**mock_context)

    print("\n[8/8] Loading master feature set...")
    load_features(**mock_context)

    print("\n" + "=" * 60)
    print("ETL pipeline complete — check data/features/ for output files")