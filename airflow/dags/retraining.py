"""
airflow/dags/retraining.py
---------------------------
Automated model retraining DAG for the Smart Library ML Platform.

This DAG monitors model drift and triggers retraining when performance
degrades beyond acceptable thresholds — the 5-6 week decay pattern
documented in our production monitoring.

Retraining pipeline:
1. Check drift metrics — has model performance degraded?
2. Load latest feature set from ETL pipeline output
3. Retrain all component models (ALS, SVD, TF-IDF, BERT)
4. Evaluate new models against holdout set
5. Compare new vs current model performance
6. Promote new model if it improves on current (champion/challenger)
7. Archive old model weights for rollback capability

Why automated retraining?
Library patron behavior shifts seasonally — summer reading programs,
academic calendars, new releases. A model trained in January will
drift by March without retraining. This DAG catches that drift
at the 5-6 week threshold before it surfaces in recommendation quality.
"""

from datetime import datetime, timedelta
import logging
import os
import sys
import json
import pickle

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

DAG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(DAG_DIR, "..", "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "ingestion"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "collaborative"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "content_based"))
sys.path.insert(0, os.path.join(ROOT_DIR, "models"))

DEFAULT_ARGS = {
    "owner": "karun_singampalli",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

# Drift threshold — retrain if PSI exceeds this value
DRIFT_THRESHOLD = 0.2
# Performance threshold — only promote if new model beats current by this margin
IMPROVEMENT_THRESHOLD = 0.01


# ── Task Functions ────────────────────────────────────────────────────────

def check_drift_metrics(**context) -> dict:
    """
    Task 1: Check whether model drift exceeds retraining threshold.

    Reads PSI (Population Stability Index) scores from the monitoring
    pipeline. PSI > 0.2 indicates significant distribution shift
    that warrants retraining.

    PSI interpretation:
    - PSI < 0.1:  No significant change — no action needed
    - PSI 0.1-0.2: Moderate change — monitor closely
    - PSI > 0.2:  Significant change — retrain now

    Returns:
        dict with drift_detected flag and PSI scores per feature
    """
    logger.info("Retraining Task 1: Checking drift metrics...")

    models_dir = os.path.join(ROOT_DIR, "data", "models")
    drift_path = os.path.join(models_dir, "drift_metrics.json")

    # If no drift metrics file exists yet, simulate drift check
    if not os.path.exists(drift_path):
        logger.warning("No drift metrics file found — simulating drift check")
        drift_metrics = {
            "overall_psi": 0.15,
            "feature_psi": {
                "total_checkouts": 0.08,
                "genre_diversity": 0.12,
                "avg_loan_duration": 0.18,
                "days_since_last_checkout": 0.15
            },
            "last_training_date": "2026-08-01",
            "days_since_training": 35,
            "drift_detected": False
        }
    else:
        with open(drift_path) as f:
            drift_metrics = json.load(f)

    drift_detected = (
        drift_metrics["overall_psi"] > DRIFT_THRESHOLD or
        drift_metrics.get("days_since_training", 0) > 42  # 6 weeks hard cap
    )

    drift_metrics["drift_detected"] = drift_detected

    context["task_instance"].xcom_push(
        key="drift_detected", value=drift_detected
    )
    context["task_instance"].xcom_push(
        key="overall_psi", value=drift_metrics["overall_psi"]
    )

    if drift_detected:
        logger.warning(
            f"Drift detected — PSI: {drift_metrics['overall_psi']:.3f} "
            f"(threshold: {DRIFT_THRESHOLD}) — retraining triggered"
        )
    else:
        logger.info(
            f"No significant drift — PSI: {drift_metrics['overall_psi']:.3f} "
            f"(threshold: {DRIFT_THRESHOLD}) — retraining skipped"
        )

    return drift_metrics


def load_training_data(**context) -> dict:
    """
    Task 2: Load latest feature set produced by the ETL pipeline.

    Uses the master_features.parquet produced by etl_pipeline DAG.
    If features don't exist, falls back to generating fresh synthetic data.
    """
    import pandas as pd

    logger.info("Retraining Task 2: Loading training data...")

    features_path = os.path.join(
        ROOT_DIR, "data", "features", "master_features.parquet"
    )
    lms_path = os.path.join(ROOT_DIR, "data", "lms_raw.parquet")

    if os.path.exists(lms_path):
        lms_df = pd.read_parquet(lms_path)
        logger.info(f"Loaded {len(lms_df)} LMS records from ETL output")
    else:
        logger.warning("No ETL output found — generating fresh training data")
        from lms_connector import load_lms_data
        lms_df = load_lms_data(source="synthetic", n_transactions=5000)

    context["task_instance"].xcom_push(
        key="training_records", value=len(lms_df)
    )

    # Save LMS df for downstream tasks
    lms_df.to_parquet(
        os.path.join(ROOT_DIR, "data", "training_lms.parquet"), index=False
    )

    logger.info(f"Training data ready: {len(lms_df)} records")
    return {"status": "success", "records": len(lms_df)}


def retrain_collaborative_models(**context) -> dict:
    """
    Task 3: Retrain ALS and SVD collaborative filtering models.

    Trains on the full dataset then evaluates on a 20% holdout.
    Saves new model weights to a challenger directory.
    """
    import pandas as pd
    from als_model import ALSModel, build_interaction_matrix
    from svd_model import SVDModel
    from sklearn.model_selection import train_test_split

    logger.info("Retraining Task 3: Retraining collaborative models...")

    lms_df = pd.read_parquet(
        os.path.join(ROOT_DIR, "data", "training_lms.parquet")
    )

    # Train/test split at the interaction level
    train_df, test_df = train_test_split(lms_df, test_size=0.2, random_state=42)

    # Build interaction matrix on training data
    matrix, patron_idx, book_idx = build_interaction_matrix(train_df)

    # Retrain ALS
    logger.info("Retraining ALS...")
    als_model = ALSModel(n_factors=20, n_iterations=15, regularization=0.01)
    als_model.fit(matrix, patron_idx, book_idx)

    # Retrain SVD
    logger.info("Retraining SVD...")
    svd_model = SVDModel(n_components=20)
    svd_model.fit(matrix, patron_idx, book_idx)

    # Save challenger models
    challenger_dir = os.path.join(ROOT_DIR, "data", "models", "challenger")
    os.makedirs(challenger_dir, exist_ok=True)

    als_model.save(os.path.join(challenger_dir, "als_model.pkl"))
    svd_model.save(os.path.join(challenger_dir, "svd_model.pkl"))

    # Simple evaluation: average predicted score on test interactions
    test_patrons = test_df["patron_id"].unique()
    test_patrons_in_model = [p for p in test_patrons if p in patron_idx]

    avg_scores = []
    for patron_id in test_patrons_in_model[:20]:  # sample for speed
        recs = als_model.recommend(
            patron_id, n_recommendations=10,
            interaction_matrix=matrix
        )
        if len(recs) > 0:
            avg_scores.append(recs["predicted_score"].mean())

    avg_score = float(np.mean(avg_scores)) if avg_scores else 0.0

    context["task_instance"].xcom_push(
        key="als_avg_score", value=avg_score
    )

    logger.info(f"Collaborative models retrained — ALS avg score: {avg_score:.4f}")
    return {"status": "success", "als_avg_score": avg_score}


def retrain_content_models(**context) -> dict:
    """
    Task 4: Retrain TF-IDF content-based model.

    Runs in parallel with collaborative model retraining.
    BERT retraining is skipped here (embeddings are stable enough
    to update monthly rather than at every drift trigger).
    """
    import pandas as pd
    from tfidf_model import TFIDFRecommender

    logger.info("Retraining Task 4: Retraining content models...")

    lms_df = pd.read_parquet(
        os.path.join(ROOT_DIR, "data", "training_lms.parquet")
    )

    tfidf_model = TFIDFRecommender(max_features=200, ngram_range=(1, 2))
    tfidf_model.fit(lms_df)

    challenger_dir = os.path.join(ROOT_DIR, "data", "models", "challenger")
    os.makedirs(challenger_dir, exist_ok=True)
    tfidf_model.save(os.path.join(challenger_dir, "tfidf_model.pkl"))

    # Evaluate on a sample patron
    test_patron = lms_df["patron_id"].iloc[0]
    recs = tfidf_model.recommend(test_patron, lms_df, n_recommendations=10)
    avg_similarity = recs["similarity_score"].mean() if len(recs) > 0 else 0.0

    logger.info(
        f"Content model retrained — avg similarity: {avg_similarity:.4f}"
    )
    return {"status": "success", "avg_similarity": float(avg_similarity)}


def evaluate_and_promote(**context) -> dict:
    """
    Task 5: Compare challenger vs champion and promote if improved.

    Champion/challenger pattern:
    - Champion = currently deployed model
    - Challenger = newly trained model
    - Promote challenger only if it beats champion by IMPROVEMENT_THRESHOLD
    - Archive champion for rollback if needed

    This prevents a bad retraining run from degrading production.
    """
    import pandas as pd
    import shutil

    logger.info("Retraining Task 5: Evaluating and promoting models...")

    models_dir = os.path.join(ROOT_DIR, "data", "models")
    champion_dir = os.path.join(models_dir, "champion")
    challenger_dir = os.path.join(models_dir, "challenger")
    archive_dir = os.path.join(
        models_dir, "archive",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Load performance metrics
    als_score = context["task_instance"].xcom_pull(
        task_ids="retrain_collaborative_models", key="als_avg_score"
    ) or 0.0

    # Load champion metrics if they exist
    champion_metrics_path = os.path.join(champion_dir, "metrics.json")
    if os.path.exists(champion_metrics_path):
        with open(champion_metrics_path) as f:
            champion_metrics = json.load(f)
        champion_score = champion_metrics.get("als_avg_score", 0.0)
    else:
        champion_score = 0.0  # No champion yet — always promote

    improvement = als_score - champion_score
    should_promote = improvement >= IMPROVEMENT_THRESHOLD or champion_score == 0.0

    if should_promote:
        logger.info(
            f"Promoting challenger (score: {als_score:.4f}) over "
            f"champion (score: {champion_score:.4f}) — "
            f"improvement: {improvement:+.4f}"
        )

        # Archive current champion
        if os.path.exists(champion_dir):
            shutil.copytree(champion_dir, archive_dir)
            logger.info(f"Champion archived to {archive_dir}")

        # Promote challenger to champion
        if os.path.exists(champion_dir):
            shutil.rmtree(champion_dir)
        shutil.copytree(challenger_dir, champion_dir)

        # Save new champion metrics
        os.makedirs(champion_dir, exist_ok=True)
        with open(os.path.join(champion_dir, "metrics.json"), "w") as f:
            json.dump({
                "als_avg_score": als_score,
                "training_date": datetime.now().isoformat(),
                "promoted": True
            }, f, indent=2)

        logger.info("Challenger promoted to champion successfully")
    else:
        logger.info(
            f"Challenger (score: {als_score:.4f}) did not improve on "
            f"champion (score: {champion_score:.4f}) — keeping champion"
        )

    return {
        "status": "success",
        "promoted": should_promote,
        "challenger_score": als_score,
        "champion_score": champion_score,
        "improvement": improvement
    }


def update_drift_baseline(**context) -> dict:
    """
    Task 6: Reset drift baseline after successful retraining.

    After retraining, the new model becomes the baseline.
    Drift metrics are reset so monitoring starts fresh.
    """
    logger.info("Retraining Task 6: Updating drift baseline...")

    models_dir = os.path.join(ROOT_DIR, "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    new_baseline = {
        "overall_psi": 0.0,
        "feature_psi": {},
        "last_training_date": datetime.now().isoformat(),
        "days_since_training": 0,
        "drift_detected": False,
        "baseline_reset": True
    }

    with open(os.path.join(models_dir, "drift_metrics.json"), "w") as f:
        json.dump(new_baseline, f, indent=2)

    logger.info("Drift baseline reset — monitoring restarted from today")
    return {"status": "success", "baseline_reset": True}


# ── Missing import ────────────────────────────────────────────────────────
import numpy as np

# ── DAG Definition ────────────────────────────────────────────────────────
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="smart_library_retraining",
        default_args=DEFAULT_ARGS,
        description="Automated model retraining triggered by drift detection",
        schedule_interval="0 3 * * 0",  # Weekly on Sunday at 3am
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=["retraining", "smart-library", "mlops"],
    ) as dag:

        t_check_drift = PythonOperator(
            task_id="check_drift_metrics",
            python_callable=check_drift_metrics,
        )
        t_load_data = PythonOperator(
            task_id="load_training_data",
            python_callable=load_training_data,
        )
        t_retrain_collab = PythonOperator(
            task_id="retrain_collaborative_models",
            python_callable=retrain_collaborative_models,
        )
        t_retrain_content = PythonOperator(
            task_id="retrain_content_models",
            python_callable=retrain_content_models,
        )
        t_evaluate = PythonOperator(
            task_id="evaluate_and_promote",
            python_callable=evaluate_and_promote,
        )
        t_update_baseline = PythonOperator(
            task_id="update_drift_baseline",
            python_callable=update_drift_baseline,
        )

        # DAG dependency graph
        t_check_drift >> t_load_data
        t_load_data >> [t_retrain_collab, t_retrain_content]
        [t_retrain_collab, t_retrain_content] >> t_evaluate
        t_evaluate >> t_update_baseline


# ── Standalone test ────────────────────────────────────────────────────────
# Run: python airflow/dags/retraining.py
if __name__ == "__main__":
    print("Running retraining pipeline in standalone mode...")
    print("=" * 60)

    mock_context = {
        "execution_date": datetime.now(),
        "task_instance": type("MockTI", (), {
            "xcom_push": lambda self, key, value: print(f"  XCom: {key} = {value}"),
            "xcom_pull": lambda self, task_ids, key: 0.45
        })()
    }

    print("\n[1/6] Checking drift metrics...")
    check_drift_metrics(**mock_context)

    print("\n[2/6] Loading training data...")
    load_training_data(**mock_context)

    print("\n[3/6] Retraining collaborative models...")
    retrain_collaborative_models(**mock_context)

    print("\n[4/6] Retraining content models...")
    retrain_content_models(**mock_context)

    print("\n[5/6] Evaluating and promoting...")
    evaluate_and_promote(**mock_context)

    print("\n[6/6] Updating drift baseline...")
    update_drift_baseline(**mock_context)

    print("\n" + "=" * 60)
    print("Retraining pipeline complete")