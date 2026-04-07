from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MSME Credit AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
              radial-gradient(circle at top left, rgba(12, 200, 190, 0.18), transparent 24%),
              radial-gradient(circle at top right, rgba(72, 140, 255, 0.16), transparent 22%),
              linear-gradient(180deg, #08111c 0%, #0b1523 55%, #07111d 100%);
            color: #ecf5ff;
        }
        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero-card {
            padding: 28px 30px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(12, 22, 38, 0.92), rgba(16, 31, 53, 0.82));
            border: 1px solid rgba(140, 173, 202, 0.15);
            box-shadow: 0 30px 80px rgba(2, 10, 18, 0.45);
            margin-bottom: 20px;
        }
        .eyebrow {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(49, 196, 191, 0.12);
            border: 1px solid rgba(49, 196, 191, 0.24);
            color: #9ae7e4;
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .hero-title {
            font-size: 3rem;
            line-height: 0.95;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
        }
        .hero-copy {
            color: #91a7bb;
            font-size: 1rem;
            line-height: 1.8;
            max-width: 860px;
            margin-top: 16px;
            margin-bottom: 0;
        }
        .metric-card {
            padding: 20px 22px;
            border-radius: 22px;
            background: rgba(13, 24, 39, 0.84);
            border: 1px solid rgba(132, 171, 202, 0.14);
            box-shadow: 0 24px 60px rgba(3, 10, 18, 0.35);
        }
        .metric-label {
            color: #90a8bd;
            font-size: 0.88rem;
            margin-bottom: 10px;
        }
        .metric-value {
            color: #ecf5ff;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        div[data-testid="stMetric"] {
            background: rgba(13, 24, 39, 0.84);
            border: 1px solid rgba(132, 171, 202, 0.14);
            padding: 18px 18px 10px 18px;
            border-radius: 22px;
        }
        div[data-testid="stMetricLabel"] label {
            color: #90a8bd !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ecf5ff !important;
        }
        div[data-testid="stSidebar"] {
            background: rgba(10, 18, 31, 0.96);
            border-right: 1px solid rgba(132, 171, 202, 0.1);
        }
        .decision-box {
            padding: 22px;
            border-radius: 22px;
            background: rgba(13, 24, 39, 0.84);
            border: 1px solid rgba(132, 171, 202, 0.14);
            box-shadow: 0 24px 60px rgba(3, 10, 18, 0.35);
        }
        .decision-score {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(49, 196, 191, 0.12);
            border: 1px solid rgba(49, 196, 191, 0.24);
            color: #9ae7e4;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(path: str) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def load_overview() -> dict:
    return api_get("/analytics/overview")


@st.cache_data(ttl=60)
def load_importance() -> list[dict]:
    return api_get("/analytics/feature-importance")


def hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">AI Underwriting Command Center</div>
          <h1 class="hero-title">AI-Powered MSME Credit Scoring & Risk Analytics Platform</h1>
          <p class="hero-copy">
            Modern fintech-grade decisioning for MSME lenders, blending automated credit scoring,
            explainable risk analytics, and approval recommendations in one executive dashboard.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(kpis: dict) -> None:
    cols = st.columns(4)
    cards = [
        ("Applications", kpis["applications"]),
        ("Average Score", kpis["avg_credit_score"]),
        ("Approval Ready", f'{kpis["approval_ready_pct"]}%'),
        ("High Risk", f'{kpis["high_risk_pct"]}%'),
    ]
    for col, (label, value) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_charts(overview: dict, importance: list[dict]) -> None:
    col1, col2 = st.columns([1.7, 1.1])

    with col1:
        trend_df = pd.DataFrame(overview["vintage_trend"])
        fig = px.area(
            trend_df,
            x="label",
            y="score",
            title="Score Momentum by MSME Vintage",
            color_discrete_sequence=["#31c4bf"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f2",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_df = pd.DataFrame(overview["risk_distribution"])
        fig = px.pie(
            risk_df,
            names="label",
            values="count",
            hole=0.62,
            title="Risk Classification",
            color="label",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f2",
            margin=dict(l=10, r=10, t=50, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns([1.1, 1.7])

    with col3:
        score_band_df = pd.DataFrame(overview["score_band_distribution"])
        fig = px.bar(
            score_band_df,
            x="label",
            y="count",
            title="Score Bands",
            color_discrete_sequence=["#59a8ff"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f2",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        importance_df = pd.DataFrame(importance[:8]).sort_values("importance")
        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Explainable AI Feature Importance",
            color_discrete_sequence=["#31c4bf"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f2",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Importance",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    revenue_df = pd.DataFrame(overview["revenue_vs_score"])
    fig = px.bar(
        revenue_df,
        x="label",
        y="score",
        title="Revenue Tier vs Credit Score",
        color_discrete_sequence=["#7c8cff"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce8f2",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Revenue Segment",
        yaxis_title="Avg Credit Score",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_table(overview: dict) -> None:
    st.subheader("Top Performing Applicants")
    table_df = pd.DataFrame(overview["top_applicants"])
    table_df["annual_revenue"] = table_df["annual_revenue"].map(lambda value: f"Rs {value:,.0f}")
    table_df["default_probability"] = table_df["default_probability"].map(lambda value: f"{value * 100:.1f}%")
    table_df["net_profit_margin"] = table_df["net_profit_margin"].map(lambda value: f"{value * 100:.1f}%")
    table_df["digital_transactions_ratio"] = table_df["digital_transactions_ratio"].map(
        lambda value: f"{value * 100:.0f}%"
    )
    st.dataframe(
        table_df.rename(
            columns={
                "annual_revenue": "Revenue",
                "credit_score": "Score",
                "default_probability": "Default Prob.",
                "net_profit_margin": "Margin",
                "digital_transactions_ratio": "Digital Ratio",
                "sector_risk_score": "Sector Risk",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def sidebar_form() -> dict:
    st.sidebar.markdown("## Credit Decision Studio")
    st.sidebar.caption("Score a new MSME profile in real time")

    return {
        "annual_revenue": st.sidebar.number_input("Annual Revenue", min_value=100000.0, value=4200000.0),
        "net_profit_margin": st.sidebar.number_input("Net Profit Margin", min_value=-0.5, max_value=0.7, value=0.16),
        "cash_flow_coverage": st.sidebar.number_input("Cash Flow Coverage", min_value=0.0, max_value=10.0, value=1.7),
        "debt_to_income": st.sidebar.number_input("Debt to Income", min_value=0.0, max_value=2.0, value=0.31),
        "gst_compliance_score": st.sidebar.number_input("GST Compliance Score", min_value=0.0, max_value=100.0, value=88.0),
        "bank_balance_volatility": st.sidebar.number_input("Bank Balance Volatility", min_value=0.0, max_value=1.0, value=0.18),
        "emi_bounce_ratio": st.sidebar.number_input("EMI Bounce Ratio", min_value=0.0, max_value=1.0, value=0.03),
        "invoice_payment_delay_days": st.sidebar.number_input("Invoice Delay Days", min_value=0.0, max_value=180.0, value=11.0),
        "digital_transactions_ratio": st.sidebar.number_input("Digital Transactions Ratio", min_value=0.0, max_value=1.0, value=0.74),
        "bureau_inquiries_6m": st.sidebar.number_input("Bureau Inquiries 6M", min_value=0, max_value=25, value=1),
        "sector_risk_score": st.sidebar.number_input("Sector Risk Score", min_value=0.0, max_value=100.0, value=32.0),
        "vintage_months": st.sidebar.number_input("Business Vintage Months", min_value=1, max_value=600, value=54),
        "geo_stability_score": st.sidebar.number_input("Geo Stability Score", min_value=0.0, max_value=100.0, value=78.0),
        "owner_credit_history_years": st.sidebar.number_input("Owner Credit History Years", min_value=0.0, max_value=40.0, value=8.0),
    }


def decision_panel(payload: dict) -> None:
    st.subheader("Underwriting Decision")
    left, right = st.columns([0.9, 1.1])

    with left:
        if st.button("Generate Credit Decision", use_container_width=True, type="primary"):
            try:
                st.session_state["prediction"] = api_post("/predict", payload)
            except requests.RequestException as exc:
                st.session_state["prediction_error"] = str(exc)

        if st.button("Retrain Model", use_container_width=True):
            try:
                api_post("/train")
                load_overview.clear()
                load_importance.clear()
                st.session_state["prediction_error"] = ""
                st.success("Model retrained successfully.")
            except requests.RequestException as exc:
                st.session_state["prediction_error"] = str(exc)

    with right:
        prediction = st.session_state.get("prediction")
        prediction_error = st.session_state.get("prediction_error", "")

        if prediction_error:
            st.error(prediction_error)

        if prediction:
            st.markdown(
                f"""
                <div class="decision-box">
                  <div style="color:#90a8bd;font-size:0.9rem;">Credit Score</div>
                  <div class="decision-score">{prediction["credit_score"]}</div>
                  <div class="pill">{prediction["risk_classification"]} Risk</div>
                  <div class="pill">{prediction["loan_recommendation"]}</div>
                  <div style="margin-top:14px;color:#90a8bd;">Default Probability</div>
                  <div style="font-size:1.2rem;font-weight:700;">{prediction["default_probability"] * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            driver_cols = st.columns(2)
            driver_cols[0].markdown("#### Positive Drivers")
            for item in prediction["top_positive_drivers"]:
                driver_cols[0].markdown(f"- {item.replace('_', ' ').title()}")
            driver_cols[1].markdown("#### Negative Drivers")
            for item in prediction["top_negative_drivers"]:
                driver_cols[1].markdown(f"- {item.replace('_', ' ').title()}")
        else:
            st.info("Generate a credit decision to see score, risk class, and approval recommendation.")


def main() -> None:
    inject_styles()
    hero()

    try:
        overview = load_overview()
        importance = load_importance()
    except requests.RequestException:
        st.error(
            "Unable to reach the FastAPI backend. Start the backend service and ensure API_BASE_URL is correct."
        )
        return

    metric_cards(overview["portfolio_kpis"])
    st.markdown("")
    build_charts(overview, importance)
    payload = sidebar_form()
    decision_panel(payload)
    render_table(overview)


if __name__ == "__main__":
    main()
