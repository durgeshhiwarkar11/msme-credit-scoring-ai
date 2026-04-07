from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATASET_PATH


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def generate_synthetic_msme_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_revenue = rng.lognormal(mean=15.1, sigma=0.55, size=rows)
    net_profit_margin = _clip(rng.normal(0.14, 0.08, rows), -0.2, 0.45)
    cash_flow_coverage = _clip(rng.normal(1.5, 0.55, rows), 0.2, 4.5)
    debt_to_income = _clip(rng.normal(0.45, 0.22, rows), 0.05, 1.6)
    gst_compliance_score = _clip(rng.normal(79, 12, rows), 35, 99)
    bank_balance_volatility = _clip(rng.normal(0.24, 0.11, rows), 0.02, 0.85)
    emi_bounce_ratio = _clip(rng.beta(1.6, 12, rows), 0.0, 0.85)
    invoice_payment_delay_days = _clip(rng.normal(17, 11, rows), 0, 90)
    digital_transactions_ratio = _clip(rng.beta(5, 2.3, rows), 0.05, 0.99)
    bureau_inquiries_6m = rng.poisson(1.4, rows)
    sector_risk_score = _clip(rng.normal(42, 18, rows), 10, 95)
    vintage_months = _clip(rng.normal(46, 26, rows), 6, 180)
    geo_stability_score = _clip(rng.normal(72, 14, rows), 25, 98)
    owner_credit_history_years = _clip(rng.normal(7.5, 4.1, rows), 0.5, 25)

    raw_risk = (
        1.25 * debt_to_income
        + 1.2 * emi_bounce_ratio
        + 0.015 * invoice_payment_delay_days
        + 0.012 * sector_risk_score
        + 0.08 * bureau_inquiries_6m
        + 0.9 * bank_balance_volatility
        - 0.9 * net_profit_margin
        - 0.5 * cash_flow_coverage
        - 0.0075 * gst_compliance_score
        - 0.65 * digital_transactions_ratio
        - 0.004 * vintage_months
        - 0.006 * geo_stability_score
        - 0.03 * owner_credit_history_years
        - 0.00000008 * annual_revenue
        + rng.normal(0, 0.22, rows)
    )

    default_probability = 1.0 / (1.0 + np.exp(-raw_risk))
    is_high_risk = (default_probability > 0.5).astype(int)
    credit_score = _clip((900 - default_probability * 600).round(), 300, 900).astype(int)

    return pd.DataFrame(
        {
            "annual_revenue": annual_revenue.round(2),
            "net_profit_margin": net_profit_margin.round(4),
            "cash_flow_coverage": cash_flow_coverage.round(4),
            "debt_to_income": debt_to_income.round(4),
            "gst_compliance_score": gst_compliance_score.round(2),
            "bank_balance_volatility": bank_balance_volatility.round(4),
            "emi_bounce_ratio": emi_bounce_ratio.round(4),
            "invoice_payment_delay_days": invoice_payment_delay_days.round(2),
            "digital_transactions_ratio": digital_transactions_ratio.round(4),
            "bureau_inquiries_6m": bureau_inquiries_6m.astype(int),
            "sector_risk_score": sector_risk_score.round(2),
            "vintage_months": vintage_months.round().astype(int),
            "geo_stability_score": geo_stability_score.round(2),
            "owner_credit_history_years": owner_credit_history_years.round(2),
            "default_probability": default_probability.round(4),
            "is_high_risk": is_high_risk.astype(int),
            "credit_score": credit_score,
        }
    )


def ensure_dataset(path: Path = DATASET_PATH, rows: int = 2500) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    dataset = generate_synthetic_msme_data(rows=rows)
    dataset.to_csv(path, index=False)
    return dataset
