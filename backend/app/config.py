from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATASET_PATH = DATA_DIR / "msme_credit_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "credit_risk_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.joblib"

FEATURE_COLUMNS = [
    "annual_revenue",
    "net_profit_margin",
    "cash_flow_coverage",
    "debt_to_income",
    "gst_compliance_score",
    "bank_balance_volatility",
    "emi_bounce_ratio",
    "invoice_payment_delay_days",
    "digital_transactions_ratio",
    "bureau_inquiries_6m",
    "sector_risk_score",
    "vintage_months",
    "geo_stability_score",
    "owner_credit_history_years",
]
