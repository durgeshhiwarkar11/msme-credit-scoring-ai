from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import (
    analytics_overview,
    feature_importance,
    load_model_bundle,
    predict,
    reload_model_bundle,
)
from app.schemas import MSMEApplication, PredictionResponse, TrainingResponse
from ml.train import train_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_bundle()
    yield


app = FastAPI(
    title="AI-Powered MSME Credit Scoring & Risk Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    _, metadata = load_model_bundle()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": metadata["model_type"],
        "auc": metadata["auc"],
    }


@app.get("/analytics/overview")
def get_analytics_overview():
    return analytics_overview()


@app.get("/analytics/feature-importance")
def get_feature_importance():
    return feature_importance()


@app.post("/predict", response_model=PredictionResponse)
def predict_credit(application: MSMEApplication):
    return predict(application)


@app.post("/train", response_model=TrainingResponse)
def retrain_model():
    result = train_pipeline()
    reload_model_bundle()
    return {
        "status": "retrained",
        "rows": result["dataset_rows"],
        "model_type": result["model_type"],
        "auc": result["auc"],
    }
