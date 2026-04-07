from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
from app.schemas import MSMEApplication
from ml.generate_data import ensure_dataset
from ml.train import train_pipeline


DRIVER_DIRECTIONS = {
    "annual_revenue": "higher",
    "net_profit_margin": "higher",
    "cash_flow_coverage": "higher",
    "debt_to_income": "lower",
    "gst_compliance_score": "higher",
    "bank_balance_volatility": "lower",
    "emi_bounce_ratio": "lower",
    "invoice_payment_delay_days": "lower",
    "digital_transactions_ratio": "higher",
    "bureau_inquiries_6m": "lower",
    "sector_risk_score": "lower",
    "vintage_months": "higher",
    "geo_stability_score": "higher",
    "owner_credit_history_years": "higher",
}


def _ensure_artifacts() -> None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_pipeline()


@lru_cache(maxsize=1)
def load_model_bundle():
    _ensure_artifacts()
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


def reload_model_bundle():
    load_model_bundle.cache_clear()
    return load_model_bundle()


def _risk_classification(default_probability: float) -> str:
    if default_probability < 0.28:
        return "Low"
    if default_probability < 0.55:
        return "Medium"
    return "High"


def _loan_recommendation(score: int, risk_classification: str) -> str:
    if score >= 720 and risk_classification == "Low":
        return "Approve"
    if score >= 620 and risk_classification != "High":
        return "Review"
    return "Decline"


def _driver_lists(payload: dict, metadata: dict) -> tuple[list[str], list[str]]:
    means = metadata["feature_means"]
    importance = metadata["feature_importance"]
    impacts = []

    for feature in FEATURE_COLUMNS:
        mean_value = means[feature]
        value = payload[feature]
        normalized_gap = (value - mean_value) / (abs(mean_value) + 1e-9)
        signed_impact = normalized_gap * importance.get(feature, 0.0)
        if DRIVER_DIRECTIONS[feature] == "lower":
            signed_impact *= -1
        impacts.append((feature, signed_impact))

    positive = [name for name, _ in sorted(impacts, key=lambda item: item[1], reverse=True)[:3]]
    negative = [name for name, _ in sorted(impacts, key=lambda item: item[1])[:3]]
    return positive, negative


def predict(application: MSMEApplication) -> dict:
    model, metadata = load_model_bundle()
    payload = application.model_dump()
    features = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    default_probability = float(model.predict_proba(features)[0][1])
    credit_score = int(np.clip(round(900 - default_probability * 600), 300, 900))
    risk_classification = _risk_classification(default_probability)
    recommendation = _loan_recommendation(credit_score, risk_classification)
    positive, negative = _driver_lists(payload, metadata)

    return {
        "credit_score": credit_score,
        "default_probability": round(default_probability, 4),
        "risk_classification": risk_classification,
        "loan_recommendation": recommendation,
        "top_positive_drivers": positive,
        "top_negative_drivers": negative,
    }


def analytics_overview() -> dict:
    _, metadata = load_model_bundle()
    dataset = ensure_dataset()

    score_band_distribution = [
        {"label": "Excellent", "count": int((dataset["credit_score"] >= 780).sum())},
        {
            "label": "Good",
            "count": int(
                ((dataset["credit_score"] >= 680) & (dataset["credit_score"] < 780)).sum()
            ),
        },
        {
            "label": "Watchlist",
            "count": int(
                ((dataset["credit_score"] >= 580) & (dataset["credit_score"] < 680)).sum()
            ),
        },
        {"label": "High Risk", "count": int((dataset["credit_score"] < 580).sum())},
    ]

    risk_distribution = [
        {"label": "Low", "count": int((dataset["default_probability"] < 0.28).sum())},
        {
            "label": "Medium",
            "count": int(
                (
                    (dataset["default_probability"] >= 0.28)
                    & (dataset["default_probability"] < 0.55)
                ).sum()
            ),
        },
        {"label": "High", "count": int((dataset["default_probability"] >= 0.55).sum())},
    ]

    revenue_vs_score = (
        dataset.assign(revenue_bucket=pd.qcut(dataset["annual_revenue"], 6, duplicates="drop"))
        .groupby("revenue_bucket", observed=False)["credit_score"]
        .mean()
        .reset_index()
    )
    revenue_vs_score["revenue_bucket"] = revenue_vs_score["revenue_bucket"].astype(str)

    vintage_trend = (
        dataset.assign(vintage_bucket=pd.cut(dataset["vintage_months"], bins=[0, 12, 24, 36, 60, 120, 240]))
        .groupby("vintage_bucket", observed=False)["credit_score"]
        .mean()
        .reset_index()
    )
    vintage_trend["vintage_bucket"] = vintage_trend["vintage_bucket"].astype(str)

    top_applicants = (
        dataset.sort_values("credit_score", ascending=False)
        .head(8)[
            [
                "annual_revenue",
                "credit_score",
                "default_probability",
                "net_profit_margin",
                "digital_transactions_ratio",
                "sector_risk_score",
            ]
        ]
        .copy()
    )
    top_applicants["default_probability"] = top_applicants["default_probability"].round(3)

    return {
        "portfolio_kpis": {
            "model_type": metadata["model_type"],
            "auc": metadata["auc"],
            "applications": int(len(dataset)),
            "avg_credit_score": int(dataset["credit_score"].mean().round()),
            "approval_ready_pct": round(float((dataset["credit_score"] >= 720).mean() * 100), 2),
            "high_risk_pct": round(float((dataset["default_probability"] >= 0.55).mean() * 100), 2),
        },
        "score_band_distribution": score_band_distribution,
        "risk_distribution": risk_distribution,
        "revenue_vs_score": revenue_vs_score.rename(
            columns={"revenue_bucket": "label", "credit_score": "score"}
        ).to_dict(orient="records"),
        "vintage_trend": vintage_trend.rename(
            columns={"vintage_bucket": "label", "credit_score": "score"}
        ).to_dict(orient="records"),
        "top_applicants": top_applicants.to_dict(orient="records"),
    }


def feature_importance() -> list[dict]:
    _, metadata = load_model_bundle()
    return [
        {"feature": feature, "importance": round(float(weight), 4)}
        for feature, weight in metadata["feature_importance"].items()
    ]
