from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(data_path)
    return dataset


def engineer_features(dataset: pd.DataFrame) -> pd.DataFrame:
    engineered = dataset.copy()
    revenue = engineered["annual_revenue"].replace(0, np.nan)
    engineered["cashflow_margin"] = engineered["cash_flow"] / revenue
    engineered["utilization_gap"] = 100 - engineered["credit_utilization"]
    engineered["history_utilization_interaction"] = (
        engineered["loan_history"] * engineered["credit_utilization"]
    )
    engineered["cashflow_revenue_gap"] = engineered["annual_revenue"] - engineered["cash_flow"]
    return engineered


def compute_feature_importance(model, feature_names: list[str]) -> list[dict]:
    estimator = model.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        importances = np.abs(estimator.coef_[0])

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .to_dict(orient="records")
    )
    return ranking


def try_compute_shap(best_pipeline: Pipeline, X_train: pd.DataFrame) -> list[dict]:
    try:
        import shap  # type: ignore
    except Exception:
        return []

    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        classifier = best_pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(X_train.head(100))
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer = shap.Explainer(classifier, transformed)
            values = explainer(transformed).values

        mean_abs = np.abs(values).mean(axis=0)
        shap_summary = (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
        return shap_summary
    except Exception:
        return []


def train_and_save_model(data_path: str | Path = BASE_DIR / "data.csv") -> dict:
    dataset = load_dataset(data_path)
    dataset = engineer_features(dataset)

    feature_columns = [
        "annual_revenue",
        "cash_flow",
        "loan_history",
        "credit_utilization",
        "cashflow_margin",
        "utilization_gap",
        "history_utilization_interaction",
        "cashflow_revenue_gap",
    ]
    target_column = "approved"

    X = dataset[feature_columns]
    y = dataset[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_columns)]
    )

    model_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=6, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    leaderboard = []
    best_model_name = None
    best_pipeline = None
    best_accuracy = -1.0
    best_roc_auc = -1.0

    for name, estimator in model_candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)
        leaderboard.append({"model": name, "accuracy": accuracy, "roc_auc": roc_auc})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    feature_importance = compute_feature_importance(best_pipeline, feature_columns)
    shap_summary = try_compute_shap(best_pipeline, X_train)

    artifacts = {
        "pipeline": best_pipeline,
        "feature_columns": feature_columns,
        "metrics": {
            "best_model": best_model_name,
            "accuracy": best_accuracy,
            "roc_auc": best_roc_auc,
            "leaderboard": leaderboard,
        },
        "feature_importance": feature_importance,
        "thresholds": {"low": 0.72, "medium": 0.48},
        "shap_summary": shap_summary,
    }

    joblib.dump(artifacts, MODEL_PATH)
    return artifacts


if __name__ == "__main__":
    train_and_save_model()
