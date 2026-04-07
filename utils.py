from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_INPUT_COLUMNS = [
    "annual_revenue",
    "cash_flow",
    "loan_history",
    "credit_utilization",
]

MODEL_FEATURE_LABELS = {
    "annual_revenue": "Annual Revenue",
    "cash_flow": "Cash Flow",
    "loan_history": "Loan History",
    "credit_utilization": "Credit Utilization",
    "cashflow_margin": "Cash Flow Margin",
    "utilization_gap": "Utilization Headroom",
    "history_utilization_interaction": "History x Utilization",
    "cashflow_revenue_gap": "Revenue-Cash Flow Gap",
}


def load_artifacts(model_path: str | Path) -> dict:
    return joblib.load(model_path)


def validate_dataset_columns(dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    missing_columns = [column for column in BASE_INPUT_COLUMNS if column not in dataset.columns]
    if missing_columns:
        st.error(
            f"Uploaded dataset is missing required columns: {', '.join(missing_columns)}"
        )
        st.stop()


def add_engineered_features(dataset: pd.DataFrame) -> pd.DataFrame:
    enriched = dataset.copy()
    revenue = enriched["annual_revenue"].replace(0, np.nan)
    enriched["cashflow_margin"] = enriched["cash_flow"] / revenue
    enriched["utilization_gap"] = 100 - enriched["credit_utilization"]
    enriched["history_utilization_interaction"] = (
        enriched["loan_history"] * enriched["credit_utilization"]
    )
    enriched["cashflow_revenue_gap"] = enriched["annual_revenue"] - enriched["cash_flow"]
    return enriched


def derive_credit_score(approval_probability: np.ndarray, input_df: pd.DataFrame) -> np.ndarray:
    base_score = 300 + (approval_probability * 600)
    utilization_penalty = np.clip(input_df["credit_utilization"].to_numpy() - 65, 0, None) * 1.7
    weak_history_penalty = np.clip(5 - input_df["loan_history"].to_numpy(), 0, None) * 14
    negative_cashflow_penalty = np.where(input_df["cash_flow"].to_numpy() < 0, 40, 0)
    score = base_score - utilization_penalty - weak_history_penalty - negative_cashflow_penalty
    return np.clip(score, 300, 900).round(0)


def assign_risk_category(probabilities: np.ndarray, thresholds: dict) -> np.ndarray:
    low_threshold = thresholds["low"]
    medium_threshold = thresholds["medium"]
    categories = np.where(
        probabilities >= low_threshold,
        "Low",
        np.where(probabilities >= medium_threshold, "Medium", "High"),
    )
    return categories


def build_confidence(probabilities: np.ndarray) -> np.ndarray:
    return (0.5 + np.abs(probabilities - 0.5)).clip(0.5, 1.0)


def predict_credit_outcome(artifacts: dict, applicant_df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_engineered_features(applicant_df)
    model_input = enriched[artifacts["feature_columns"]]
    probabilities = artifacts["pipeline"].predict_proba(model_input)[:, 1]
    scores = derive_credit_score(probabilities, applicant_df)
    risk_categories = assign_risk_category(probabilities, artifacts["thresholds"])
    confidence = build_confidence(probabilities)
    approvals = np.where(probabilities >= 0.55, "Approved", "Review")

    return pd.DataFrame(
        {
            "credit_score": scores,
            "risk_category": risk_categories,
            "approval_probability": probabilities,
            "approval_status": approvals,
            "confidence_score": confidence,
        }
    )


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_score(value: float) -> str:
    return f"{int(round(value))}"


def explain_prediction(applicant: dict, result: dict) -> str:
    reasons = []
    if applicant["cash_flow"] < 0:
        reasons.append("negative cash flow is pressuring repayment capacity")
    elif applicant["cash_flow"] < applicant["annual_revenue"] * 0.12:
        reasons.append("cash flow is modest relative to annual revenue")
    else:
        reasons.append("healthy cash flow supports repayment strength")

    if applicant["credit_utilization"] > 65:
        reasons.append("credit utilization is elevated")
    else:
        reasons.append("credit utilization remains under control")

    if applicant["loan_history"] >= 7:
        reasons.append("strong loan history improves confidence")
    elif applicant["loan_history"] <= 3:
        reasons.append("limited loan history increases uncertainty")

    return (
        f"The applicant is classified as {result['risk_category']} risk because "
        + ", ".join(reasons[:3])
        + "."
    )


def create_risk_distribution_chart(risk_series: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
    counts = risk_series.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    colors = ["#34d399", "#fbbf24", "#f87171"]
    ax.bar(counts.index, counts.values, color=colors, width=0.55)
    ax.set_title("Predicted Risk Bands", color="#f8fafc", fontsize=13)
    ax.set_ylabel("Applicants", color="#cbd5e1")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return fig


def create_loan_trend_chart(dataset: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
    rolling = (
        dataset["approval_probability"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
        .rolling(window=5, min_periods=1)
        .mean()
    )
    ax.plot(rolling.index + 1, rolling.values, color="#38bdf8", linewidth=2.5)
    ax.fill_between(rolling.index + 1, rolling.values, color="#0ea5e9", alpha=0.18)
    ax.set_title("Approval Trendline", color="#f8fafc", fontsize=13)
    ax.set_xlabel("Portfolio Slice", color="#cbd5e1")
    ax.set_ylabel("Approval Probability", color="#cbd5e1")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return fig


def build_portfolio_overview(dataset: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
    labels = ["High Confidence", "Needs Review"]
    sizes = [
        int((dataset["approval_probability"] >= 0.55).sum()),
        int((dataset["approval_probability"] < 0.55).sum()),
    ]
    ax.pie(
        sizes,
        labels=labels,
        colors=["#10b981", "#fb7185"],
        autopct="%1.0f%%",
        textprops={"color": "#e2e8f0"},
        wedgeprops={"linewidth": 1, "edgecolor": "#0f172a"},
    )
    ax.set_title("Approval Confidence Mix", color="#f8fafc", fontsize=13)
    return fig


def create_feature_importance_chart(feature_importance: list[dict]):
    chart_df = pd.DataFrame(feature_importance).head(8).sort_values("importance")
    chart_df["label"] = chart_df["feature"].map(MODEL_FEATURE_LABELS).fillna(chart_df["feature"])
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0f172a")
    ax.barh(chart_df["label"], chart_df["importance"], color="#60a5fa")
    ax.set_title("Top Model Drivers", color="#f8fafc", fontsize=13)
    ax.set_xlabel("Importance", color="#cbd5e1")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return fig


def create_risk_segmentation_chart(dataset: pd.DataFrame):
    working = dataset.copy()
    bins = pd.cut(
        working["credit_utilization"],
        bins=[0, 30, 60, 100],
        labels=["0-30%", "31-60%", "61-100%"],
        include_lowest=True,
    )
    grouped = working.groupby(bins)["loan_history"].mean().fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
    ax.plot(grouped.index.astype(str), grouped.values, marker="o", color="#f59e0b", linewidth=2.5)
    ax.set_title("Risk Segmentation by Utilization", color="#f8fafc", fontsize=13)
    ax.set_ylabel("Average Loan History", color="#cbd5e1")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return fig


def build_dashboard_insights(dataset: pd.DataFrame) -> list[str]:
    insights = []
    high_risk_share = (dataset["predicted_risk"] == "High").mean()
    low_risk_share = (dataset["predicted_risk"] == "Low").mean()
    avg_utilization = dataset["credit_utilization"].mean()
    median_cash_flow = dataset["cash_flow"].median()

    insights.append(f"Low-risk MSMEs account for {low_risk_share * 100:.1f}% of the current portfolio.")
    insights.append(f"High-risk exposure stands at {high_risk_share * 100:.1f}% based on model inference.")
    insights.append(f"Average credit utilization is {avg_utilization:.1f}%, which is a key leading stress indicator.")
    insights.append(f"Median cash flow is {median_cash_flow:,.0f}, supporting quick peer comparison.")
    return insights


def build_lender_insights(dataset: pd.DataFrame) -> list[str]:
    review_rate = (dataset["approval_probability"] < 0.55).mean() * 100
    strong_score_share = (dataset["predicted_score"] >= 720).mean() * 100
    return [
        f"{strong_score_share:.1f}% of applicants fall into a lender-friendly score band above 720.",
        f"{review_rate:.1f}% of cases may need manual underwriting review based on current thresholds.",
        "Use the feature importance panel to explain approval and decline patterns to credit committees.",
    ]


def build_msme_guidance(applicant: dict, result: dict) -> list[str]:
    guidance = []
    if applicant["credit_utilization"] > 60:
        guidance.append("Reduce credit utilization below 60% to improve the next lending decision.")
    if applicant["cash_flow"] < applicant["annual_revenue"] * 0.12:
        guidance.append("Improve operating cash flow conversion to strengthen repayment capacity.")
    if applicant["loan_history"] < 5:
        guidance.append("Build stronger repayment history through smaller facilities before larger borrowing.")
    if not guidance:
        guidance.append("This profile already shows healthy underwriting signals for most MSME loan products.")
    guidance.append(
        f"Current approval probability is {format_percentage(result['approval_probability'])}; small balance-sheet improvements can move this higher."
    )
    return guidance


def summarize_uploaded_data(dataset: pd.DataFrame) -> dict:
    return {
        "rows": int(len(dataset)),
        "avg_revenue": float(dataset["annual_revenue"].mean()),
        "avg_cash_flow": float(dataset["cash_flow"].mean()),
        "avg_utilization": float(dataset["credit_utilization"].mean()),
    }
