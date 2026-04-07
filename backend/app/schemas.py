from typing import Literal

from pydantic import BaseModel, Field


class MSMEApplication(BaseModel):
    annual_revenue: float = Field(..., ge=100000)
    net_profit_margin: float = Field(..., ge=-0.5, le=0.7)
    cash_flow_coverage: float = Field(..., ge=0.0, le=10.0)
    debt_to_income: float = Field(..., ge=0.0, le=2.0)
    gst_compliance_score: float = Field(..., ge=0.0, le=100.0)
    bank_balance_volatility: float = Field(..., ge=0.0, le=1.0)
    emi_bounce_ratio: float = Field(..., ge=0.0, le=1.0)
    invoice_payment_delay_days: float = Field(..., ge=0.0, le=180.0)
    digital_transactions_ratio: float = Field(..., ge=0.0, le=1.0)
    bureau_inquiries_6m: int = Field(..., ge=0, le=25)
    sector_risk_score: float = Field(..., ge=0.0, le=100.0)
    vintage_months: int = Field(..., ge=1, le=600)
    geo_stability_score: float = Field(..., ge=0.0, le=100.0)
    owner_credit_history_years: float = Field(..., ge=0.0, le=40.0)


class PredictionResponse(BaseModel):
    credit_score: int
    default_probability: float
    risk_classification: Literal["Low", "Medium", "High"]
    loan_recommendation: Literal["Approve", "Review", "Decline"]
    top_positive_drivers: list[str]
    top_negative_drivers: list[str]


class TrainingResponse(BaseModel):
    status: str
    rows: int
    model_type: str
    auc: float
