from __future__ import annotations

from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import ARTIFACTS_DIR, FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
from ml.generate_data import ensure_dataset

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def _build_model():
    if XGBClassifier is not None:
        return (
            XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            ),
            "XGBoost",
        )
    return (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=3,
            random_state=42,
        ),
        "RandomForest",
    )


def train_pipeline() -> dict[str, Any]:
    dataset = ensure_dataset()
    X = dataset[FEATURE_COLUMNS]
    y = dataset["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, model_type = _build_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    metadata = {
        "model_type": model_type,
        "auc": float(round(auc, 4)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_means": X.mean().to_dict(),
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "dataset_rows": int(len(dataset)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)
    return metadata
