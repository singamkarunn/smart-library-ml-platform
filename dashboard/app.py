"""
dashboard/app.py
-----------------
Streamlit patron analytics dashboard for the Smart Library ML Platform.

Provides library staff with:
1. Patron behavior overview — borrowing patterns, engagement metrics
2. Book catalog analytics — genre trends, popularity, seasonal patterns
3. Recommendation explorer — test recommendations for any patron
4. Model health monitor — drift metrics, pipeline status, latency
5. Live API tester — call the recommendation API directly from the dashboard

Run: streamlit run dashboard/app.py
Then visit: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta

# ── Path setup ────────────────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(DASHBOARD_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "ingestion"))
sys.path.insert(0, os.path.join(ROOT_DIR, "features"))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Library ML Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8faff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #0f2044;
    }
    .metric-lbl {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-healthy { color: #059669; font-weight: 600; }
    .status-warning { color: #d97706; font-weight: 600; }
    .status-critical { color: #dc2626; font-weight: 600; }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #0f2044;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loader ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """
    Loads LMS, POS, and activity data.
    Uses ETL output if available, otherwise generates synthetic data.
    """
    data_dir = os.path.join(ROOT_DIR, "data")

    lms_path = os.path.join(data_dir, "lms_raw.parquet")
    pos_path = os.path.join(data_dir, "pos_raw.parquet")
    activity_path = os.path.join(data_dir, "activity_raw.parquet")

    if os.path.exists(lms_path):
        lms_df = pd.read_parquet(lms_path)
        pos_df = pd.read_parquet(pos_path) if os.path.exists(pos_path) else None
        activity_df = pd.read_parquet(activity_path) if os.path.exists(activity_path) else None
    else:
        from lms_connector import load_lms_data
        from pos_connector import load_pos_data
        from kafka_consumer import load_activity_events
        lms_df = load_lms_data(source="synthetic", n_transactions=5000)
        pos_df = load_pos_data(source="synthetic", n_transactions=2000)
        activity_df = load_activity_events(source="synthetic", n_events=2000)

    return lms_df, pos_df, activity_df


@st.cache_data(ttl=60)
def load_drift_report():
    """Loads latest drift monitoring report."""
    report_path = os.path.join(ROOT_DIR, "data", "models", "drift_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f2044/ffffff?text=Smart+Library", width=200)
    st.markdown("### Navigation")

    page = st.radio(
        "Select View",
        ["📊 Overview", "👥 Patron Analytics",
         "📚 Book Analytics", "🎯 Recommendation Explorer",
         "🔧 Model Health", "⚡ Live API Tester"]
    )

    st.markdown("---")
    st.markdown("**Data Controls**")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**API Connection**")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.status_code == 200:
            health = resp.json()
            st.success(f"✅ API Connected")
            st.caption(
                f"Patrons: {health['n_patrons_in_model']} | "
                f"Books: {health['n_books_in_model']}"
            )
        else:
            st.warning("⚠️ API Responding (non-200)")
    except Exception:
        st.error("❌ API Offline — Start with: `cd api && python main.py`")

    st.markdown("---")
    st.caption("Smart Library ML Platform v1.0")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ── Load Data ─────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    lms_df, pos_df, activity_df = load_data()
    drift_report = load_drift_report()

lms_df["checkout_date"] = pd.to_datetime(lms_df["checkout_date"])
lms_df["return_date"] = pd.to_datetime(lms_df["return_date"])


# ════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📚 Smart Library ML Platform")
    st.markdown("**Patron Analytics & Recommendation Intelligence Dashboard**")
    st.markdown("---")

    # ── Top metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{lms_df['patron_id'].nunique():,}</div>
            <div class="metric-lbl">Active Patrons</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{lms_df['book_id'].nunique():,}</div>
            <div class="metric-lbl">Books in Catalog</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(lms_df):,}</div>
            <div class="metric-lbl">Total Checkouts</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        avg_loan = lms_df["loan_duration_days"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{avg_loan:.1f}d</div>
            <div class="metric-lbl">Avg Loan Duration</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        overdue_rate = (lms_df["loan_status"] == "overdue").mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{overdue_rate:.1%}</div>
            <div class="metric-lbl">Overdue Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Checkouts Over Time</div>',
                    unsafe_allow_html=True)
        daily = lms_df.groupby(
            lms_df["checkout_date"].dt.to_period("W").dt.start_time
        ).size().reset_index(name="checkouts")
        fig = px.area(
            daily, x="checkout_date", y="checkouts",
            color_discrete_sequence=["#2563eb"],
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="", yaxis_title="Weekly Checkouts"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Genre Distribution</div>',
                    unsafe_allow_html=True)
        genre_counts = lms_df["genre"].value_counts().reset_index()
        genre_counts.columns = ["genre", "count"]
        fig = px.bar(
            genre_counts, x="count", y="genre",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis_title="", xaxis_title="Checkouts"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Model status ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Status</div>',
                unsafe_allow_html=True)

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    models = [
        ("ALS", "Collaborative", "Weight: 0.35", "✅"),
        ("SVD", "Collaborative", "Weight: 0.25", "✅"),
        ("TF-IDF", "Content-Based", "Weight: 0.20", "✅"),
        ("BERT", "Content-Based", "Weight: 0.20", "✅"),
    ]

    for col, (name, mtype, weight, status) in zip(
        [model_col1, model_col2, model_col3, model_col4], models
    ):
        with col:
            st.metric(label=f"{status} {name}", value=mtype, delta=weight)


# ════════════════════════════════════════════════════════════════
# PAGE: PATRON ANALYTICS
# ════════════════════════════════════════════════════════════════
elif page == "👥 Patron Analytics":
    st.title("👥 Patron Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Membership Distribution</div>',
                    unsafe_allow_html=True)
        membership = lms_df.drop_duplicates("patron_id")
        if "patron_membership_type" in membership.columns:
            mem_counts = membership["patron_membership_type"].value_counts()
            fig = px.pie(
                values=mem_counts.values,
                names=mem_counts.index,
                color_discrete_sequence=["#0f2044", "#2563eb", "#93c5fd"],
                template="plotly_white"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Age Group Distribution</div>',
                    unsafe_allow_html=True)
        if "patron_age_group" in lms_df.columns:
            age_counts = lms_df.drop_duplicates("patron_id")[
                "patron_age_group"
            ].value_counts().sort_index()
            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                color_discrete_sequence=["#2563eb"],
                template="plotly_white",
                labels={"x": "Age Group", "y": "Patrons"}
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Top patrons
    st.markdown('<div class="section-header">Most Active Patrons</div>',
                unsafe_allow_html=True)
    top_patrons = lms_df.groupby("patron_id").agg(
        checkouts=("book_id", "count"),
        unique_books=("book_id", "nunique"),
        avg_duration=("loan_duration_days", "mean"),
        overdue_count=("loan_status", lambda x: (x == "overdue").sum())
    ).sort_values("checkouts", ascending=False).head(10).round(2)

    st.dataframe(top_patrons, use_container_width=True)

    # Borrowing heatmap by day of week and hour
    st.markdown('<div class="section-header">Borrowing Patterns — Day of Week</div>',
                unsafe_allow_html=True)
    lms_df["day_of_week"] = lms_df["checkout_date"].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_counts = lms_df["day_of_week"].value_counts().reindex(day_order)

    fig = px.bar(
        x=day_order,
        y=day_counts.values,
        color_discrete_sequence=["#2563eb"],
        template="plotly_white",
        labels={"x": "Day of Week", "y": "Checkouts"}
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: BOOK ANALYTICS
# ════════════════════════════════════════════════════════════════
elif page == "📚 Book Analytics":
    st.title("📚 Book Catalog Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Most Borrowed Books</div>',
                    unsafe_allow_html=True)
        top_books = lms_df.groupby(["book_id", "genre"]).agg(
            borrows=("patron_id", "count"),
            unique_patrons=("patron_id", "nunique"),
            avg_renewals=("times_renewed", "mean")
        ).sort_values("borrows", ascending=False).head(15).round(2)
        st.dataframe(top_books, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Genre Trends Over Time</div>',
                    unsafe_allow_html=True)
        lms_df["month"] = lms_df["checkout_date"].dt.to_period("M").dt.start_time
        genre_time = lms_df.groupby(
            ["month", "genre"]
        ).size().reset_index(name="checkouts")

        top_genres = lms_df["genre"].value_counts().head(5).index.tolist()
        genre_time_top = genre_time[genre_time["genre"].isin(top_genres)]

        fig = px.line(
            genre_time_top, x="month", y="checkouts",
            color="genre",
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="", yaxis_title="Monthly Checkouts"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal patterns
    st.markdown('<div class="section-header">Seasonal Borrowing Patterns</div>',
                unsafe_allow_html=True)
    lms_df["season"] = lms_df["checkout_date"].dt.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    season_genre = lms_df.groupby(
        ["season", "genre"]
    ).size().reset_index(name="checkouts")

    fig = px.bar(
        season_genre, x="season", y="checkouts",
        color="genre",
        barmode="group",
        template="plotly_white",
        category_orders={"season": ["Spring", "Summer", "Fall", "Winter"]}
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Season", yaxis_title="Checkouts"
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATION EXPLORER
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendation Explorer":
    st.title("🎯 Recommendation Explorer")
    st.markdown("Test the hybrid recommendation engine for any patron.")
    st.markdown("---")

    patron_ids = sorted(lms_df["patron_id"].unique().tolist())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_patron = st.selectbox("Select Patron", patron_ids)
    with col2:
        n_recs = st.slider("Recommendations", 5, 20, 10)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        get_recs = st.button("🎯 Get Recommendations", use_container_width=True)

    if get_recs:
        with st.spinner("Generating recommendations..."):
            start = time.time()
            try:
                resp = requests.post(
                    f"{api_url}/recommend",
                    json={"patron_id": selected_patron,
                          "n_recommendations": n_recs},
                    timeout=60
                )
                elapsed = (time.time() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    recs = data["recommendations"]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Recommendations", len(recs))
                    col2.metric("Inference Time", f"{data['inference_time_ms']:.0f}ms")
                    col3.metric("Models Used", data["n_models_used"])

                    st.markdown("### Top Recommendations")
                    recs_df = pd.DataFrame(recs)
                    recs_df["score_bar"] = recs_df["combined_score"]

                    st.dataframe(
                        recs_df[["rank","book_id","genre","combined_score"]],
                        use_container_width=True
                    )

                    # Genre distribution of recommendations
                    genre_dist = recs_df["genre"].value_counts()
                    fig = px.bar(
                        x=genre_dist.index, y=genre_dist.values,
                        color_discrete_sequence=["#2563eb"],
                        template="plotly_white",
                        labels={"x": "Genre", "y": "Count"},
                        title="Genre Distribution of Recommendations"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error: {resp.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Start it with: `cd api && python main.py`")

    # Patron borrowing history
    st.markdown("### Borrowing History")
    patron_history = lms_df[lms_df["patron_id"] == selected_patron][[
        "book_id", "book_title", "genre",
        "checkout_date", "loan_duration_days", "loan_status"
    ]].sort_values("checkout_date", ascending=False)
    st.dataframe(patron_history.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: MODEL HEALTH
# ════════════════════════════════════════════════════════════════
elif page == "🔧 Model Health":
    st.title("🔧 Model Health Monitor")
    st.markdown("---")

    if drift_report:
        status = drift_report["overall_status"]
        status_class = {
            "OK": "status-healthy",
            "WARNING": "status-warning",
            "CRITICAL": "status-critical"
        }.get(status, "status-warning")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Status", status)
        col2.metric("Overall PSI", f"{drift_report['overall_psi']:.4f}")
        col3.metric("Days Since Baseline", drift_report["days_since_baseline"])
        col4.metric(
            "Retraining Recommended",
            "YES" if drift_report["retraining_recommended"] else "NO"
        )

        st.markdown("### Feature Drift (PSI)")
        drift_data = []
        for feature, result in drift_report["feature_drift"].items():
            drift_data.append({
                "Feature": feature,
                "PSI": result["psi"],
                "Status": result["status"],
                "Mean Shift": result["mean_shift"]
            })
        if drift_data:
            drift_df = pd.DataFrame(drift_data)
            fig = px.bar(
                drift_df, x="Feature", y="PSI",
                color="Status",
                color_discrete_map={
                    "OK": "#059669",
                    "WARNING": "#d97706",
                    "CRITICAL": "#dc2626"
                },
                template="plotly_white"
            )
            fig.add_hline(y=0.10, line_dash="dash",
                          line_color="orange", annotation_text="Warning (0.10)")
            fig.add_hline(y=0.20, line_dash="dash",
                          line_color="red", annotation_text="Critical (0.20)")
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Prediction Score Drift")
        if drift_report.get("prediction_drift"):
            pd_result = drift_report["prediction_drift"]
            col1, col2, col3 = st.columns(3)
            col1.metric("KL Divergence", f"{pd_result['kl_divergence']:.4f}")
            col2.metric("Baseline Mean Score", f"{pd_result['baseline_score_mean']:.4f}")
            col3.metric("Current Mean Score", f"{pd_result['current_score_mean']:.4f}")
    else:
        st.info(
            "No drift report found. Run the drift detector first:\n\n"
            "`python monitoring/drift_detector.py`"
        )

    # Model weights visualization
    st.markdown("### Model Weight Distribution")
    weights = {"ALS": 0.35, "SVD": 0.25, "TF-IDF": 0.20, "BERT": 0.20}
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        color_discrete_sequence=["#0f2044", "#2563eb", "#60a5fa", "#93c5fd"],
        template="plotly_white"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: LIVE API TESTER
# ════════════════════════════════════════════════════════════════
elif page == "⚡ Live API Tester":
    st.title("⚡ Live API Tester")
    st.markdown("Test all API endpoints directly from the dashboard.")
    st.markdown("---")

    endpoint = st.selectbox(
        "Select Endpoint",
        ["GET /health", "GET /model/info",
         "POST /recommend", "POST /similar-books", "POST /explain"]
    )

    if endpoint == "GET /health":
        if st.button("Send Request"):
            with st.spinner("Calling API..."):
                try:
                    resp = requests.get(f"{api_url}/health", timeout=5)
                    st.json(resp.json())
                    st.caption(f"Status: {resp.status_code} | "
                               f"Time: {resp.elapsed.total_seconds()*1000:.1f}ms")
                except Exception as e:
                    st.error(str(e))

    elif endpoint == "GET /model/info":
        if st.button("Send Request"):
            with st.spinner("Calling API..."):
                try:
                    resp = requests.get(f"{api_url}/model/info", timeout=5)
                    st.json(resp.json())
                except Exception as e:
                    st.error(str(e))

    elif endpoint == "POST /recommend":
        patron_id = st.text_input("patron_id", value="P00001")
        n_recs = st.number_input("n_recommendations", value=10, min_value=1, max_value=50)
        exclude_seen = st.checkbox("exclude_seen", value=True)

        if st.button("Send Request"):
            with st.spinner("Calling API..."):
                try:
                    payload = {
                        "patron_id": patron_id,
                        "n_recommendations": int(n_recs),
                        "exclude_seen": exclude_seen
                    }
                    st.code(json.dumps(payload, indent=2), language="json")
                    resp = requests.post(
                        f"{api_url}/recommend",
                        json=payload, timeout=60
                    )
                    st.json(resp.json())
                    st.caption(f"Status: {resp.status_code}")
                except Exception as e:
                    st.error(str(e))

    elif endpoint == "POST /similar-books":
        book_id = st.text_input("book_id", value="B00001")
        n_similar = st.number_input("n_similar", value=10, min_value=1, max_value=50)
        method = st.selectbox("method", ["tfidf", "bert"])

        if st.button("Send Request"):
            with st.spinner("Calling API..."):
                try:
                    payload = {
                        "book_id": book_id,
                        "n_similar": int(n_similar),
                        "method": method
                    }
                    resp = requests.post(
                        f"{api_url}/similar-books",
                        json=payload, timeout=30
                    )
                    st.json(resp.json())
                except Exception as e:
                    st.error(str(e))

    elif endpoint == "POST /explain":
        patron_id = st.text_input("patron_id", value="P00001")
        book_id = st.text_input("book_id", value="B00001")

        if st.button("Send Request"):
            with st.spinner("Calling API..."):
                try:
                    payload = {"patron_id": patron_id, "book_id": book_id}
                    resp = requests.post(
                        f"{api_url}/explain",
                        json=payload, timeout=30
                    )
                    st.json(resp.json())
                except Exception as e:
                    st.error(str(e))