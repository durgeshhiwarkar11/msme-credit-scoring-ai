# AI-Powered MSME Credit Scoring & Risk Analytics Platform

A production-style Python and Streamlit platform for MSME credit scoring, lender risk analytics, and explainable underwriting.

## What This App Does

- Predicts MSME creditworthiness from:
  - `annual_revenue`
  - `cash_flow`
  - `loan_history`
  - `credit_utilization`
- Generates:
  - Credit score from `300` to `900`
  - Risk category: `Low`, `Medium`, `High`
  - Approval probability
  - Approval status and confidence score
- Provides:
  - Dashboard KPIs
  - Risk distribution and trend charts
  - Feature importance and explainable AI
  - CSV upload for portfolio analysis
  - Basic login gate for protected access

## Default Project Structure

- `app.py` - main Streamlit application
- `model.py` - model training and persistence
- `utils.py` - helper functions and charts
- `data.csv` - sample MSME dataset
- `model.pkl` - generated trained model
- `requirements.txt` - dependencies
- `README.md` - setup and usage instructions

## Machine Learning

The root app compares the following scikit-learn models and automatically selects the best one based on validation accuracy:

- Logistic Regression
- Random Forest
- Gradient Boosting

The selected model is saved with `joblib` and reused for inference.

## Run The Main App

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app trains automatically on first run if `model.pkl` does not exist.

## Demo Login

By default, use:

- Username: `admin`
- Password: `msme123`

You can override these with environment variables:

```bash
set APP_USERNAME=myuser
set APP_PASSWORD=mypassword
```

## Optional Bonus API

This repository also includes an optional FastAPI backend under `backend/` and a separate API-driven dashboard under `dashboard/`.

### Run The Optional Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run The Optional API Dashboard

```bash
cd dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## API Endpoints

- `GET /health`
- `GET /analytics/overview`
- `GET /analytics/feature-importance`
- `POST /predict`
- `POST /train`

## Deployment

### Streamlit Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Cloud app.
3. Set the entry point to `app.py`.
4. Ensure `requirements.txt` is included.
5. Add `APP_USERNAME` and `APP_PASSWORD` as secrets if needed.

## Notes

- The dataset is synthetic and intended for demos and prototyping.
- SHAP explainability is shown when the package is available and compatible with the selected model.
- For production lending, you would still need model governance, compliance review, and live data integrations.
# msme-credit-scoring-ai
AI-Powered MSME Credit Scoring And Risk Analytics Platform
