import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import MODEL_PATH, train_and_save_model
from utils import (
    BASE_INPUT_COLUMNS,
    build_dashboard_insights,
    build_lender_insights,
    build_msme_guidance,
    build_portfolio_overview,
    create_feature_importance_chart,
    create_loan_trend_chart,
    create_risk_distribution_chart,
    create_risk_segmentation_chart,
    explain_prediction,
    format_percentage,
    format_score,
    load_artifacts,
    predict_credit_outcome,
    summarize_uploaded_data,
    validate_dataset_columns,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"

st.set_page_config(
    page_title="MSME Credit Scoring Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(16, 185, 129, 0.15), transparent 30%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 35%),
                linear-gradient(180deg, #07111f 0%, #0f172a 55%, #111827 100%);
            color: #e5eefc;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08101d 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        .hero-card, .info-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.26);
            backdrop-filter: blur(14px);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #f8fafc;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.75rem;
        }
        .section-copy {
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 0.75rem;
        }
        .tag {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            margin-right: 0.4rem;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #cbd5e1;
            font-size: 0.82rem;
        }
        .risk-low {
            color: #34d399;
            font-weight: 700;
        }
        .risk-medium {
            color: #fbbf24;
            font-weight: 700;
        }
        .risk-high {
            color: #f87171;
            font-weight: 700;
        }
        .login-shell {
            max-width: 500px;
            margin: 5rem auto 0 auto;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.26);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
    else:
        dataset = pd.read_csv(DATA_PATH)
    return dataset


@st.cache_resource(show_spinner=False)
def ensure_artifacts() -> dict:
    if not MODEL_PATH.exists():
        train_and_save_model(DATA_PATH)
    return load_artifacts(MODEL_PATH)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AI-Powered MSME Credit Scoring & Risk Analytics</div>
            <p class="hero-subtitle">
                A fintech-style decision intelligence workspace for banks, NBFCs, and credit teams.
                Score MSME applicants, estimate approval probability, and monitor portfolio risk from one place.
            </p>
            <div style="margin-top: 1rem;">
                <span class="tag">Banks</span>
                <span class="tag">NBFCs</span>
                <span class="tag">MSMEs</span>
                <span class="tag">FinTech APIs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_gate() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.markdown(
        """
        <div class="login-shell">
            <div class="hero-title" style="font-size:1.8rem;">Secure Access</div>
            <p class="section-copy">
                Sign in to the MSME credit intelligence workspace. Use the demo credentials shown below
                or replace them with environment variables for your own deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_username = os.getenv("APP_USERNAME", "admin")
    demo_password = os.getenv("APP_PASSWORD", "msme123")

    with st.form("login_form"):
        username = st.text_input("Username", value="")
        password = st.text_input("Password", type="password", value="")
        submitted = st.form_submit_button("Login", use_container_width=True)

    st.caption(f"Demo login: `{demo_username}` / `{demo_password}`")

    if submitted:
        if username == demo_username and password == demo_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Invalid credentials. Check the demo login or configure APP_USERNAME and APP_PASSWORD.")

    return False


def render_dashboard(artifacts: dict, dataset: pd.DataFrame) -> None:
    render_hero()
    st.write("")

    predictions = predict_credit_outcome(artifacts, dataset[BASE_INPUT_COLUMNS])
    dashboard_df = dataset.copy()
    dashboard_df["predicted_score"] = predictions["credit_score"]
    dashboard_df["predicted_risk"] = predictions["risk_category"]
    dashboard_df["approval_probability"] = predictions["approval_probability"]

    avg_score = float(np.mean(predictions["credit_score"]))
    avg_probability = float(np.mean(predictions["approval_probability"]))
    dominant_risk = dashboard_df["predicted_risk"].mode().iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Credit Score", format_score(avg_score), delta="Portfolio average")
    with col2:
        st.metric("Risk Level", dominant_risk, delta="Most common segment")
    with col3:
        st.metric("Approval Probability", format_percentage(avg_probability), delta="Predicted mean")

    chart_col1, chart_col2 = st.columns((1.15, 1))
    with chart_col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        fig = create_risk_distribution_chart(dashboard_df["predicted_risk"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)
        fig = build_portfolio_overview(dashboard_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    trend_col, insight_col = st.columns((1.2, 0.8))
    with trend_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Loan Approval Trends</div>', unsafe_allow_html=True)
        fig = create_loan_trend_chart(dashboard_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Insights For Lenders</div>', unsafe_allow_html=True)
        for insight in build_dashboard_insights(dashboard_df):
            st.markdown(f"- {insight}")
        for insight in build_lender_insights(dashboard_df):
            st.markdown(f"- {insight}")
        st.markdown("</div>", unsafe_allow_html=True)

    summary = summarize_uploaded_data(dataset)
    st.write("")
    preview_col1, preview_col2 = st.columns((0.8, 1.2))
    with preview_col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f"- Uploaded records: {summary['rows']}")
        st.markdown(f"- Average revenue: {summary['avg_revenue']:,.0f}")
        st.markdown(f"- Average cash flow: {summary['avg_cash_flow']:,.0f}")
        st.markdown(f"- Average utilization: {summary['avg_utilization']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with preview_col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Preview</div>', unsafe_allow_html=True)
        preview_df = dashboard_df[
            BASE_INPUT_COLUMNS + ["predicted_score", "predicted_risk", "approval_probability"]
        ].copy()
        preview_df["approval_probability"] = preview_df["approval_probability"].map(format_percentage)
        st.dataframe(preview_df.head(12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_page(artifacts: dict) -> None:
    render_hero()
    st.write("")

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Credit Assessment</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            annual_revenue = st.number_input("Annual Revenue", min_value=50000.0, value=850000.0, step=50000.0)
            cash_flow = st.number_input("Cash Flow", min_value=-100000.0, value=180000.0, step=10000.0)
        with col2:
            loan_history = st.slider("Loan History (0-10)", min_value=0, max_value=10, value=7)
            credit_utilization = st.slider("Credit Utilization (%)", min_value=0.0, max_value=100.0, value=38.0)

        submitted = st.form_submit_button("Predict Creditworthiness")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        applicant_df = pd.DataFrame(
            [
                {
                    "annual_revenue": annual_revenue,
                    "cash_flow": cash_flow,
                    "loan_history": loan_history,
                    "credit_utilization": credit_utilization,
                }
            ]
        )

        with st.spinner("Running ensemble risk assessment..."):
            prediction = predict_credit_outcome(artifacts, applicant_df)
            result = prediction.iloc[0].to_dict()
            explanation = explain_prediction(applicant_df.iloc[0].to_dict(), result)
            st.session_state["last_prediction"] = result

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Credit Score", format_score(result["credit_score"]))
        col2.metric("Risk Category", result["risk_category"])
        col3.metric("Approval Status", result["approval_status"])
        col4.metric("Confidence", format_percentage(result["confidence_score"]))

        badge_class = f"risk-{result['risk_category'].lower()}"
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Decision Summary</div>
                <p>Approval probability: <strong>{format_percentage(result["approval_probability"])}</strong></p>
                <p>Risk band: <span class="{badge_class}">{result["risk_category"]}</span></p>
                <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        guidance_col, audience_col = st.columns(2)
        with guidance_col:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">How To Improve Eligibility</div>', unsafe_allow_html=True)
            for item in build_msme_guidance(applicant_df.iloc[0].to_dict(), result):
                st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)
        with audience_col:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Why This Matters</div>', unsafe_allow_html=True)
            st.markdown("- Banks can automate underwriting with a transparent approval signal.")
            st.markdown("- NBFCs can speed up high-volume credit triage with confidence scores.")
            st.markdown("- MSMEs can see which financial behaviors influence funding eligibility.")
            st.markdown("- FinTech teams can reuse the same scoring logic in API workflows.")
            st.markdown("</div>", unsafe_allow_html=True)


def render_analytics_page(artifacts: dict, dataset: pd.DataFrame) -> None:
    render_hero()
    st.write("")

    metrics = artifacts["metrics"]
    leaderboard_df = pd.DataFrame(metrics["leaderboard"]).sort_values("accuracy", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", metrics["best_model"])
    col2.metric("Model Accuracy", format_percentage(metrics["accuracy"]))
    col3.metric("Validation ROC-AUC", format_percentage(metrics["roc_auc"]))

    chart_col1, chart_col2 = st.columns((1.1, 0.9))
    with chart_col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fig = create_feature_importance_chart(artifacts["feature_importance"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(
            leaderboard_df.assign(
                accuracy=lambda frame: (frame["accuracy"] * 100).round(2).astype(str) + "%",
                roc_auc=lambda frame: (frame["roc_auc"] * 100).round(2).astype(str) + "%",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    seg_col, data_col = st.columns((1, 1))
    with seg_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Segmentation</div>', unsafe_allow_html=True)
        fig = create_risk_segmentation_chart(dataset)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with data_col:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data Insights</div>', unsafe_allow_html=True)
        stats_df = dataset.describe().T[["mean", "min", "max"]].round(2)
        st.dataframe(stats_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    shap_values = artifacts.get("shap_summary")
    if shap_values:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Explainable AI Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            "SHAP is available for the trained best model. Top global drivers are ranked below based on mean absolute contribution."
        )
        shap_df = pd.DataFrame(shap_values).sort_values("importance", ascending=False)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Lender Decision Notes</div>', unsafe_allow_html=True)
    st.markdown("- Higher revenue and cash flow generally improve credit score and approval probability.")
    st.markdown("- High credit utilization and weak repayment history tend to move applicants into manual review.")
    st.markdown("- Use the leaderboard to compare model stability before promoting changes to production.")
    st.markdown("- The included backend API can expose this model to other lending systems when needed.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_about_page() -> None:
    render_hero()
    col1, col2 = st.columns((1, 1))
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Target Users</div>', unsafe_allow_html=True)
        st.markdown("- Banks and financial institutions: automate underwriting and reduce default risk.")
        st.markdown("- NBFCs: accelerate high-volume loan decisions with real-time assessment.")
        st.markdown("- MSMEs: understand financial health and improve loan readiness.")
        st.markdown("- FinTech companies: integrate scoring into lending products and partner workflows.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Bonus Capabilities</div>', unsafe_allow_html=True)
        st.markdown("- Optional SHAP summary for explainability when available in the environment.")
        st.markdown("- Optional FastAPI backend in the `backend/` folder for API-based integration.")
        st.markdown("- Basic login gate using `APP_USERNAME` and `APP_PASSWORD` environment variables.")
        st.markdown("- Streamlit Cloud deployment is documented in the project README.")
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    if not render_login_gate():
        return

    with st.spinner("Preparing credit intelligence engine..."):
        artifacts = ensure_artifacts()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Prediction", "Risk Analytics", "About"])
    uploaded_file = st.sidebar.file_uploader("Upload MSME dataset", type=["csv"])
    st.sidebar.markdown("`Dark fintech mode` is enabled by default.")
    st.sidebar.markdown("Use your own CSV with the base columns for instant portfolio analysis.")
    if st.sidebar.button("Retrain Sample Model", use_container_width=True):
        with st.spinner("Retraining model on the current dataset..."):
            train_and_save_model(DATA_PATH)
            st.cache_resource.clear()
            st.success("Sample model retrained and saved to model.pkl")
            st.rerun()
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    dataset = load_dataset(uploaded_file)
    validate_dataset_columns(dataset, artifacts["feature_columns"])

    if page == "Dashboard":
        render_dashboard(artifacts, dataset)
    elif page == "Prediction":
        render_prediction_page(artifacts)
    elif page == "Risk Analytics":
        render_analytics_page(artifacts, dataset)
    else:
        render_about_page()


if __name__ == "__main__":
    main()
