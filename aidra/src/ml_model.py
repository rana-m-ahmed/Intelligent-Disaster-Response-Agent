from __future__ import annotations

import csv
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import clone
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "Disaster-Reported-Dataset.csv")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dataset_splits")
VALIDATION_METRICS_PATH = os.path.join(MODELS_DIR, "ml_validation_metrics.json")
REQUIRED_COLUMNS = {
    "incident_id",
    "disaster_type",
    "event_type",
    "patient_age",
    "sex",
    "triage_level",
    "severity_score",
    "gcs",
    "systolic_bp",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "distance_to_hospital_km",
    "response_delay_min",
    "local_hazard_level",
    "time_since_event_hr",
    "comorbidity_index",
    "injury_mechanism",
    "survival_label",
    "risk_label",
}


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
            if os.path.exists(DATASET_PATH):
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
        rows = self._load_dataset_rows()
        X, y_survival, y_risk, feature_rows = self._generate_dataset(rows)
        train_idx, val_idx, test_idx = self._split_indices(X, y_survival, y_risk)
        self._write_split_artifacts(rows, feature_rows, train_idx, val_idx, test_idx)

        survival_models = self._candidate_models()
        risk_models = self._candidate_models()

        survival_best_name, survival_validation_reports = self._train_and_select(
            survival_models,
            X[train_idx],
            y_survival[train_idx],
            X[val_idx],
            y_survival[val_idx],
        )
        risk_best_name, risk_validation_reports = self._train_and_select(
            risk_models,
            X[train_idx],
            y_risk[train_idx],
            X[val_idx],
            y_risk[val_idx],
        )

        survival_final = clone(survival_models[survival_best_name])
        risk_final = clone(risk_models[risk_best_name])
        survival_final.fit(
            np.vstack([X[train_idx], X[val_idx]]),
            np.concatenate([y_survival[train_idx], y_survival[val_idx]]),
        )
        risk_final.fit(
            np.vstack([X[train_idx], X[val_idx]]),
            np.concatenate([y_risk[train_idx], y_risk[val_idx]]),
        )

        survival_test_reports = self._evaluate_models(
            survival_models,
            np.vstack([X[train_idx], X[val_idx]]),
            np.concatenate([y_survival[train_idx], y_survival[val_idx]]),
            X[test_idx],
            y_survival[test_idx],
        )
        risk_test_reports = self._evaluate_models(
            risk_models,
            np.vstack([X[train_idx], X[val_idx]]),
            np.concatenate([y_risk[train_idx], y_risk[val_idx]]),
            X[test_idx],
            y_risk[test_idx],
        )

        self.survival_model = survival_final
        self.risk_model = risk_final
        self.metrics["survival"] = survival_test_reports
        self.metrics["risk"] = risk_test_reports

        with open(VALIDATION_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "survival": {
                        name: report.__dict__ for name, report in survival_validation_reports.items()
                    },
                    "risk": {
                        name: report.__dict__ for name, report in risk_validation_reports.items()
                    },
                    "selected_models": {
                        "survival": survival_best_name,
                        "risk": risk_best_name,
                    },
                },
                f,
                indent=2,
            )

        with open(os.path.join(MODELS_DIR, "survival_model.pkl"), "wb") as f:
            pickle.dump(self.survival_model, f)
        with open(os.path.join(MODELS_DIR, "risk_model.pkl"), "wb") as f:
            pickle.dump(self.risk_model, f)
        with open(os.path.join(MODELS_DIR, "ml_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "survival": {
                        name: report.__dict__ for name, report in survival_test_reports.items()
                    },
                    "risk": {
                        name: report.__dict__ for name, report in risk_test_reports.items()
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

    def _load_dataset_rows(self) -> List[Dict[str, str]]:
        """Load the approved CSV dataset from disk."""
        with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError("Dataset is empty")
        return rows

    def _generate_dataset(
        self, rows: List[Dict[str, str]] | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """Load and transform the CSV dataset into the model feature matrix."""
        if rows is None:
            rows = self._load_dataset_rows()
        feature_rows: List[Dict[str, float]] = []
        features: List[List[float]] = []
        y_survival: List[int] = []
        y_risk: List[int] = []

        for row in rows:
            severity = self._safe_int(row["severity_score"])
            distance = self._safe_float(row["distance_to_hospital_km"]) / 10.0
            area_risk = self._safe_float(row["local_hazard_level"])
            time_since = self._safe_float(row["time_since_event_hr"])

            feature_row = {
                "severity": float(severity),
                "distance": float(distance),
                "area_risk": float(area_risk),
                "time_since": float(time_since),
            }
            feature_rows.append(feature_row)
            features.append(
                [
                    feature_row["severity"],
                    feature_row["distance"],
                    feature_row["area_risk"],
                    feature_row["time_since"],
                ]
            )
            y_survival.append(self._safe_int(row["survival_label"]))
            y_risk.append(self._safe_int(row["risk_label"]))

        return (
            np.asarray(features, dtype=float),
            np.asarray(y_survival, dtype=int),
            np.asarray(y_risk, dtype=int),
            feature_rows,
        )

    def _split_indices(
        self,
        X: np.ndarray,
        y_survival: np.ndarray,
        y_risk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified train, validation, and test splits."""
        indices = np.arange(len(X))
        stratify = self._combined_stratify(y_survival, y_risk)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=42,
            stratify=stratify,
        )

        temp_stratify = self._combined_stratify(y_survival[temp_idx], y_risk[temp_idx])
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=42,
            stratify=temp_stratify,
        )
        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

    @staticmethod
    def _combined_stratify(y_survival: np.ndarray, y_risk: np.ndarray) -> np.ndarray:
        combined = np.asarray([f"{int(s)}_{int(r)}" for s, r in zip(y_survival, y_risk)])
        if combined.size == 0:
            return y_survival
        _, counts = np.unique(combined, return_counts=True)
        if counts.size == 0 or counts.min() < 2:
            _, survival_counts = np.unique(y_survival, return_counts=True)
            if survival_counts.size == 0 or survival_counts.min() < 2:
                return y_risk
            return y_survival
        return combined

    def _write_split_artifacts(
        self,
        raw_rows: List[Dict[str, str]],
        feature_rows: List[Dict[str, float]],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        os.makedirs(SPLITS_DIR, exist_ok=True)
        manifest = {
            "total_rows": len(raw_rows),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "feature_contract": ["severity", "distance", "area_risk", "time_since"],
        }
        with open(os.path.join(SPLITS_DIR, "split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self._write_split_csv(os.path.join(SPLITS_DIR, "train.csv"), raw_rows, feature_rows, train_idx)
        self._write_split_csv(os.path.join(SPLITS_DIR, "validation.csv"), raw_rows, feature_rows, val_idx)
        self._write_split_csv(os.path.join(SPLITS_DIR, "test.csv"), raw_rows, feature_rows, test_idx)

    @staticmethod
    def _write_split_csv(
        path: str,
        raw_rows: List[Dict[str, str]],
        feature_rows: List[Dict[str, float]],
        indices: np.ndarray,
    ) -> None:
        fieldnames = list(raw_rows[0].keys()) + ["ml_severity", "ml_distance", "ml_area_risk", "ml_time_since"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for index in indices:
                row = dict(raw_rows[int(index)])
                features = feature_rows[int(index)]
                row.update(
                    {
                        "ml_severity": f"{features['severity']:.3f}",
                        "ml_distance": f"{features['distance']:.3f}",
                        "ml_area_risk": f"{features['area_risk']:.3f}",
                        "ml_time_since": f"{features['time_since']:.3f}",
                    }
                )
                writer.writerow(row)

    @staticmethod
    def _candidate_models() -> Dict[str, Pipeline]:
        return {
            "knn": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=7)),
                ]
            ),
            "nb": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB()),
                ]
            ),
        }

    @staticmethod
    def _safe_int(value: str) -> int:
        return int(float(value))

    @staticmethod
    def _safe_float(value: str) -> float:
        return float(value)

    @staticmethod
    def _evaluate_models(models, X_train, y_train, X_test, y_test):
        reports: Dict[str, ModelReport] = {}
        for name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            preds = fitted.predict(X_test)
            report = ModelReport(
                accuracy=accuracy_score(y_test, preds),
                precision=precision_score(y_test, preds, average="macro", zero_division=0),
                recall=recall_score(y_test, preds, average="macro", zero_division=0),
                f1=f1_score(y_test, preds, average="macro", zero_division=0),
                confusion=confusion_matrix(y_test, preds).tolist(),
            )
            reports[name] = report
        return reports

    @staticmethod
    def _train_and_select(models, X_train, y_train, X_test, y_test):
        best_model = None
        best_acc = -1.0
        reports: Dict[str, ModelReport] = {}
        for name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            preds = fitted.predict(X_test)
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
                best_model = name
        return best_model, reports
