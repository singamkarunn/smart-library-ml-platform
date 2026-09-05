"""
monitoring/drift_detector.py
-----------------------------
Production drift detection for the Smart Library ML Platform.

Monitors two types of drift:
1. Data drift (feature drift) — input data distribution has shifted
2. Concept drift (prediction drift) — model output distribution shifted

Why drift detection matters:
A recommendation model trained in September sees winter reading patterns
by December. Without drift detection, the model silently degrades —
recommendations get worse but no alert fires until a patron complains.

Our documented threshold: 5-6 weeks before significant decay.
This module catches it at 3-4 weeks, giving time to retrain before
quality degrades visibly.

Detection methods:
- PSI (Population Stability Index) for feature drift
- KL Divergence for prediction score distribution
- Chi-squared test for categorical feature drift
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from scipy import stats
from scipy.special import rel_entr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Drift Thresholds ──────────────────────────────────────────────────────
THRESHOLDS = {
    "psi_ok":           0.10,   # No action needed
    "psi_warning":      0.10,   # Monitor more closely
    "psi_critical":     0.20,   # Trigger retraining
    "kl_divergence":    0.15,   # Prediction distribution shift
    "chi2_p_value":     0.05,   # Categorical feature drift significance
    "weeks_before_retrain": 5,  # Hard cap: retrain after 5 weeks regardless
}


# ── PSI Calculator ────────────────────────────────────────────────────────
def compute_psi(baseline: np.ndarray,
                current: np.ndarray,
                n_bins: int = 10,
                feature_name: str = "") -> dict:
    """
    Computes Population Stability Index between baseline and current.

    PSI = Σ (current_pct - baseline_pct) × ln(current_pct / baseline_pct)

    Interpretation:
    - PSI < 0.10:  No significant change
    - PSI 0.10-0.20: Moderate change — monitor
    - PSI > 0.20:  Significant change — retrain

    Args:
        baseline:     Baseline distribution (from training)
        current:      Current distribution (from production)
        n_bins:       Histogram bins for comparison
        feature_name: Name for logging

    Returns:
        dict with psi score, status, and bin-level details
    """
    # Build bins from baseline percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(baseline, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    baseline_counts = np.histogram(baseline, bins=bins)[0]
    current_counts = np.histogram(current, bins=bins)[0]

    # Smooth to avoid log(0)
    baseline_pct = (baseline_counts + 1e-6) / (len(baseline) + n_bins * 1e-6)
    current_pct = (current_counts + 1e-6) / (len(current) + n_bins * 1e-6)

    bin_psi = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi = float(np.sum(bin_psi))

    # Determine status
    if psi < THRESHOLDS["psi_ok"]:
        status = "OK"
    elif psi < THRESHOLDS["psi_critical"]:
        status = "WARNING"
    else:
        status = "CRITICAL"

    result = {
        "feature": feature_name,
        "psi": round(psi, 4),
        "status": status,
        "baseline_mean": round(float(np.mean(baseline)), 4),
        "current_mean": round(float(np.mean(current)), 4),
        "mean_shift": round(float(np.mean(current) - np.mean(baseline)), 4)
    }

    if status != "OK":
        logger.warning(
            f"Drift detected — {feature_name}: "
            f"PSI={psi:.4f} ({status}), "
            f"mean shift: {result['mean_shift']:+.4f}"
        )

    return result


# ── KL Divergence for Prediction Drift ───────────────────────────────────
def compute_prediction_drift(baseline_scores: np.ndarray,
                              current_scores: np.ndarray,
                              n_bins: int = 20) -> dict:
    """
    Measures drift in model prediction score distribution using KL divergence.

    If the model is recommending the same books to everyone (popularity trap)
    or producing very low confidence scores for everyone, the prediction
    distribution shifts — this catches it.

    Args:
        baseline_scores: Score distribution from training/recent baseline
        current_scores:  Score distribution from current production traffic
        n_bins:          Histogram bins

    Returns:
        dict with KL divergence, status, and distribution statistics
    """
    bins = np.linspace(0, 1, n_bins + 1)

    baseline_hist = np.histogram(
        np.clip(baseline_scores, 0, 1), bins=bins
    )[0].astype(float)
    current_hist = np.histogram(
        np.clip(current_scores, 0, 1), bins=bins
    )[0].astype(float)

    # Normalize to probability distributions
    baseline_hist = (baseline_hist + 1e-6) / (baseline_hist.sum() + n_bins * 1e-6)
    current_hist = (current_hist + 1e-6) / (current_hist.sum() + n_bins * 1e-6)

    # KL divergence: how much information is lost using baseline dist
    # to approximate current dist
    kl_div = float(np.sum(rel_entr(current_hist, baseline_hist)))

    status = "CRITICAL" if kl_div > THRESHOLDS["kl_divergence"] else (
        "WARNING" if kl_div > THRESHOLDS["kl_divergence"] * 0.7 else "OK"
    )

    return {
        "kl_divergence": round(kl_div, 4),
        "status": status,
        "baseline_score_mean": round(float(np.mean(baseline_scores)), 4),
        "current_score_mean": round(float(np.mean(current_scores)), 4),
        "baseline_score_std": round(float(np.std(baseline_scores)), 4),
        "current_score_std": round(float(np.std(current_scores)), 4),
    }


# ── Chi-squared Test for Categorical Drift ────────────────────────────────
def compute_categorical_drift(baseline_series: pd.Series,
                               current_series: pd.Series,
                               feature_name: str = "") -> dict:
    """
    Tests for distribution shift in categorical features using chi-squared test.

    Used for genre distribution drift — if patrons suddenly borrow
    much more Science Fiction than the baseline period, the model's
    content features may need reweighting.

    Args:
        baseline_series: Baseline categorical values
        current_series:  Current categorical values
        feature_name:    Name for logging

    Returns:
        dict with chi2 statistic, p-value, and per-category shifts
    """
    # Get union of all categories
    all_categories = set(baseline_series.unique()) | set(current_series.unique())

    baseline_counts = baseline_series.value_counts()
    current_counts = current_series.value_counts()

    # Align categories
    baseline_aligned = np.array([
        baseline_counts.get(c, 0) for c in all_categories
    ], dtype=float)
    current_aligned = np.array([
        current_counts.get(c, 0) for c in all_categories
    ], dtype=float)

    # Add smoothing
    baseline_aligned += 1e-6
    current_aligned += 1e-6

    # Chi-squared test
    # H0: current distribution matches baseline
    # Reject H0 if p-value < 0.05 (significant drift)
    chi2, p_value = stats.chisquare(
        current_aligned / current_aligned.sum(),
        f_exp=baseline_aligned / baseline_aligned.sum()
    )

    drift_detected = p_value < THRESHOLDS["chi2_p_value"]
    status = "CRITICAL" if drift_detected else "OK"

    # Per-category shift
    category_shifts = {}
    baseline_pct = baseline_aligned / baseline_aligned.sum()
    current_pct = current_aligned / current_aligned.sum()
    for i, cat in enumerate(all_categories):
        shift = float(current_pct[i] - baseline_pct[i])
        if abs(shift) > 0.05:  # Only report meaningful shifts
            category_shifts[cat] = round(shift, 4)

    return {
        "feature": feature_name,
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "status": status,
        "category_shifts": category_shifts
    }


# ── Full Drift Report ─────────────────────────────────────────────────────
class DriftDetector:
    """
    Orchestrates all drift detection checks and produces a unified report.

    Usage:
        detector = DriftDetector()
        detector.set_baseline(training_lms_df)
        report = detector.run_drift_check(current_lms_df)
        detector.save_report(report)
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "models"
        )
        self.baseline_df = None
        self.baseline_set_at = None

    def set_baseline(self, lms_df: pd.DataFrame) -> None:
        """
        Sets the baseline distribution from training data.
        Call this once after training, then compare periodically.
        """
        self.baseline_df = lms_df.copy()
        self.baseline_set_at = datetime.now()
        logger.info(
            f"Drift baseline set — {len(lms_df)} records, "
            f"{lms_df['patron_id'].nunique()} patrons"
        )

    def run_drift_check(self, current_df: pd.DataFrame) -> dict:
        """
        Runs all drift checks against the stored baseline.

        Args:
            current_df: Current production data to compare against baseline

        Returns:
            Comprehensive drift report dict
        """
        if self.baseline_df is None:
            raise RuntimeError(
                "Baseline not set. Call set_baseline() first."
            )

        logger.info("Running full drift check...")
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_set_at": self.baseline_set_at.isoformat(),
            "days_since_baseline": (
                datetime.now() - self.baseline_set_at
            ).days,
            "baseline_records": len(self.baseline_df),
            "current_records": len(current_df),
            "feature_drift": {},
            "categorical_drift": {},
            "prediction_drift": None,
            "overall_status": "OK",
            "retraining_recommended": False
        }

        # ── Numeric feature drift ──────────────────────────────────────
        numeric_features = ["loan_duration_days", "times_renewed"]

        for feature in numeric_features:
            if feature in self.baseline_df.columns and \
               feature in current_df.columns:
                result = compute_psi(
                    self.baseline_df[feature].dropna().values,
                    current_df[feature].dropna().values,
                    feature_name=feature
                )
                report["feature_drift"][feature] = result

        # ── Categorical feature drift ──────────────────────────────────
        categorical_features = ["genre", "loan_status", "patron_membership_type"]

        for feature in categorical_features:
            if feature in self.baseline_df.columns and \
               feature in current_df.columns:
                result = compute_categorical_drift(
                    self.baseline_df[feature],
                    current_df[feature],
                    feature_name=feature
                )
                report["categorical_drift"][feature] = result

        # ── Prediction drift (simulated) ───────────────────────────────
        # In production: collect real prediction scores from API logs
        baseline_scores = np.random.beta(2, 3, 1000)
        current_scores = np.random.beta(2.1, 3.2, len(current_df))
        report["prediction_drift"] = compute_prediction_drift(
            baseline_scores, current_scores
        )

        # ── Overall status ─────────────────────────────────────────────
        critical_features = [
            f for f, r in report["feature_drift"].items()
            if r["status"] == "CRITICAL"
        ] + [
            f for f, r in report["categorical_drift"].items()
            if r["status"] == "CRITICAL"
        ]

        warning_features = [
            f for f, r in report["feature_drift"].items()
            if r["status"] == "WARNING"
        ]

        weeks_since_baseline = report["days_since_baseline"] / 7

        if (critical_features or
                report["prediction_drift"]["status"] == "CRITICAL" or
                weeks_since_baseline >= THRESHOLDS["weeks_before_retrain"]):
            report["overall_status"] = "CRITICAL"
            report["retraining_recommended"] = True
        elif warning_features:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "OK"

        report["critical_features"] = critical_features
        report["warning_features"] = warning_features

        # ── Overall PSI ────────────────────────────────────────────────
        psi_values = [
            r["psi"] for r in report["feature_drift"].values()
        ]
        report["overall_psi"] = round(
            float(np.mean(psi_values)) if psi_values else 0.0, 4
        )

        logger.info(
            f"Drift check complete — "
            f"status: {report['overall_status']}, "
            f"overall PSI: {report['overall_psi']}, "
            f"critical features: {len(critical_features)}"
        )

        return report

    def save_report(self, report: dict) -> str:
        """Saves drift report to disk for Airflow and Grafana to consume."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save latest report
        report_path = os.path.join(self.output_dir, "drift_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))

        # Save drift metrics (read by retraining DAG)
        drift_metrics = {
            "overall_psi": report["overall_psi"],
            "last_training_date": report["baseline_set_at"],
            "days_since_training": report["days_since_baseline"],
            "drift_detected": report["retraining_recommended"]
        }
        metrics_path = os.path.join(self.output_dir, "drift_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(drift_metrics, f, indent=2)

        logger.info(f"Drift report saved to {report_path}")
        return report_path

    def print_report(self, report: dict) -> None:
        """Prints a formatted drift report to console."""
        print("\n" + "=" * 60)
        print(f"  DRIFT DETECTION REPORT")
        print(f"  {report['timestamp']}")
        print("=" * 60)
        print(f"  Overall Status:  {report['overall_status']}")
        print(f"  Overall PSI:     {report['overall_psi']}")
        print(f"  Days since baseline: {report['days_since_baseline']}")
        print(f"  Retraining recommended: {report['retraining_recommended']}")

        print("\n── Feature Drift (PSI) ──")
        for feature, result in report["feature_drift"].items():
            icon = "✅" if result["status"] == "OK" else (
                "⚠️" if result["status"] == "WARNING" else "🚨"
            )
            print(
                f"  {icon} {feature}: PSI={result['psi']:.4f} "
                f"({result['status']}) — mean shift: {result['mean_shift']:+.4f}"
            )

        print("\n── Categorical Drift (Chi-squared) ──")
        for feature, result in report["categorical_drift"].items():
            icon = "✅" if not result["drift_detected"] else "🚨"
            print(
                f"  {icon} {feature}: p={result['p_value']:.4f} "
                f"({'DRIFT' if result['drift_detected'] else 'OK'})"
            )

        if report["prediction_drift"]:
            pd_result = report["prediction_drift"]
            icon = "✅" if pd_result["status"] == "OK" else "⚠️"
            print(f"\n── Prediction Score Drift ──")
            print(
                f"  {icon} KL divergence: {pd_result['kl_divergence']:.4f} "
                f"({pd_result['status']})"
            )

        print("\n" + "=" * 60)


# ── Quick test ────────────────────────────────────────────────────────────
# Run: python monitoring/drift_detector.py
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "ingestion"
    ))
    from lms_connector import load_lms_data

    print("Loading baseline data...")
    baseline_df = load_lms_data(source="synthetic", n_transactions=3000, seed=42)

    print("Loading current data (simulated with different seed)...")
    current_df = load_lms_data(source="synthetic", n_transactions=2800, seed=99)

    detector = DriftDetector()
    detector.set_baseline(baseline_df)

    print("\nRunning drift check...")
    report = detector.run_drift_check(current_df)
    detector.print_report(report)
    detector.save_report(report)

    print(f"\nReport saved — overall status: {report['overall_status']}")