from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class ModelReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: List[List[int]]


class MLModel:
    def __init__(self) -> None:
        """Train or load ML models for survival probability and risk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.survival_model = None
        self.risk_model = None
        self.metrics: Dict[str, Dict[str, ModelReport]] = {}
        self._fallback = False
        self._ensure_models()

    def _ensure_models(self) -> None:
        """Load existing models or train new ones, with fallback safety."""
        survival_path = os.path.join(MODELS_DIR, "survival_model.pkl")
        risk_path = os.path.join(MODELS_DIR, "risk_model.pkl")
        metrics_path = os.path.join(MODELS_DIR, "ml_metrics.json")
        # CLEANUP: avoid expensive model training on app startup unless explicitly enabled
        train_on_startup = os.getenv("ML_TRAIN_ON_STARTUP", "false").lower() == "true"
        try:
            if os.path.exists(survival_path) and os.path.exists(risk_path):
                with open(survival_path, "rb") as f:
                    self.survival_model = pickle.load(f)
                with open(risk_path, "rb") as f:
                    self.risk_model = pickle.load(f)
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self.metrics = {
                        task: {
                            name: ModelReport(**report)
                            for name, report in reports.items()
                        }
                        for task, reports in raw.items()
                    }
                return
            if train_on_startup:
                self.train_models()
            else:
                # No models on disk and training disabled — run in fallback mode.
                self._fallback = True
                return
        except Exception:
            # Fallback allows the system to run if ML loading/training fails.
            self._fallback = True

    def train_models(self) -> Dict[str, Dict[str, ModelReport]]:
        """Train and persist the best-performing models and metrics."""
        X, y_survival, y_risk = self._generate_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_survival, test_size=0.25, random_state=42
        )
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_risk, test_size=0.25, random_state=42
        )

        survival_models = {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "nb": GaussianNB(),
        }
        risk_models = {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "nb": GaussianNB(),
        }

        best_survival, survival_reports = self._train_and_select(
            survival_models, X_train, y_train, X_test, y_test
        )
        best_risk, risk_reports = self._train_and_select(
            risk_models, X_train_r, y_train_r, X_test_r, y_test_r
        )

        self.survival_model = best_survival
        self.risk_model = best_risk
        self.metrics["survival"] = survival_reports
        self.metrics["risk"] = risk_reports

        with open(os.path.join(MODELS_DIR, "survival_model.pkl"), "wb") as f:
            pickle.dump(self.survival_model, f)
        with open(os.path.join(MODELS_DIR, "risk_model.pkl"), "wb") as f:
            pickle.dump(self.risk_model, f)
        with open(os.path.join(MODELS_DIR, "ml_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "survival": {
                        name: report.__dict__ for name, report in survival_reports.items()
                    },
                    "risk": {
                        name: report.__dict__ for name, report in risk_reports.items()
                    },
                },
                f,
                indent=2,
            )
        return self.metrics

    def predict_survival(self, features: List[float]) -> float:
        """Predict survival probability in [0, 1]."""
        if self._fallback or self.survival_model is None:
            return 0.5
        proba = self.survival_model.predict_proba([features])[0][1]
        return float(max(1e-3, min(1.0, proba)))

    def predict_risk(self, features: List[float]) -> float:
        """Predict risk level normalized to [0, 1]."""
        if self._fallback or self.risk_model is None:
            return 0.5
        pred = self.risk_model.predict([features])[0]
        pred_norm = float(pred) / 2.0
        # Blend in area risk to avoid zero-risk outputs on hazardous cells.
        blended = max(pred_norm, float(features[2]))
        return float(max(0.0, min(1.0, blended)))

    def get_metrics_report(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        """Return a JSON-serializable metrics snapshot for both tasks and models."""
        return {
            task: {
                name: {
                    "accuracy": report.accuracy,
                    "precision": report.precision,
                    "recall": report.recall,
                    "f1": report.f1,
                    "confusion": report.confusion,
                }
                for name, report in reports.items()
            }
            for task, reports in self.metrics.items()
        }

    def _generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a synthetic dataset for training."""
        rng = np.random.default_rng(42)
        severity = rng.integers(0, 3, size=500)
        distance = rng.integers(1, 15, size=500)
        area_risk = rng.random(500)
        time_since = rng.integers(1, 60, size=500)
        X = np.column_stack([severity, distance, area_risk, time_since]).astype(float)
        survival_prob = (
            1.0
            - 0.3 * severity
            - 0.02 * distance
            - 0.4 * area_risk
            - 0.01 * (time_since / 10.0)
        )
        survival_prob = np.clip(survival_prob, 0.0, 1.0)
        y_survival = (survival_prob > 0.5).astype(int)
        risk_score = 0.5 * area_risk + 0.05 * severity + 0.02 * (distance / 10.0)
        y_risk = np.digitize(risk_score, bins=[0.33, 0.66])
        return X, y_survival, y_risk

    @staticmethod
    def _train_and_select(models, X_train, y_train, X_test, y_test):
        best_model = None
        best_acc = -1.0
        reports: Dict[str, ModelReport] = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = ModelReport(
                accuracy=accuracy_score(y_test, preds),
                precision=precision_score(y_test, preds, average="macro", zero_division=0),
                recall=recall_score(y_test, preds, average="macro", zero_division=0),
                f1=f1_score(y_test, preds, average="macro", zero_division=0),
                confusion=confusion_matrix(y_test, preds).tolist(),
            )
            reports[name] = report
            if report.accuracy > best_acc:
                best_acc = report.accuracy
                best_model = model
        return best_model, reports
