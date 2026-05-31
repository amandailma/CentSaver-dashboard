"""
inference.py
Prediktor CentSaver — Random Forest + Deep Learning.
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

class CentSaverRF:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X, y, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=random_state,
            class_weight="balanced", n_jobs=-1
        )
        self.model.fit(X_train_s, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_s)
        y_prob = self.model.predict_proba(X_test_s)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob),
            "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
            "X_test_s": X_test_s
        }
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model belum dilatih.")
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model belum dilatih.")
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:, 1]

    def feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    def save(self, prefix="centsaver"):
        joblib.dump(self.model, f"{prefix}_rf_model.pkl")
        joblib.dump(self.scaler, f"{prefix}_scaler.pkl")

    def load(self, prefix="centsaver"):
        self.model = joblib.load(f"{prefix}_rf_model.pkl")
        self.scaler = joblib.load(f"{prefix}_scaler.pkl")
        self.is_fitted = True

class CentSaverDL:
    """Wrapper untuk model Deep Learning Tim AI (.keras)."""
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.is_loaded = False
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self.is_loaded = True
        except Exception as e:
            print(f"[INFO] Model DL belum tersedia: {e}")

    def predict(self, X):
        if not self.is_loaded:
            return None
        X_s = self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X)
        probs = self.model.predict(X_s, verbose=0).flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        if not self.is_loaded:
            return None
        X_s = self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X)
        return self.model.predict(X_s, verbose=0).flatten()
