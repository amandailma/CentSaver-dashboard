"""
inference.py
Handles model loading, scaling, and prediction for CentSaver.
Supports both Random Forest (sklearn) and Deep Learning (TensorFlow/Keras).
"""

import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# 1. RANDOM FOREST (DS Baseline)
# ---------------------------------------------------------------------------
class RandomForestPredictor:
    def __init__(self, model_path=None):
        """
        Initialize RF predictor.
        If model_path is None, trains a new baseline model on provided data.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        if model_path:
            self.load(model_path)

    def fit(self, X, y):
        """Train baseline RF model."""
        X_s = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        self.model.fit(X_s, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Return binary predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() or .load() first.")
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def predict_proba(self, X):
        """Return probability scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() or .load() first.")
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:, 1]

    def save(self, path_prefix):
        """Save model and scaler."""
        joblib.dump(self.model, f"{path_prefix}_rf_model.pkl")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")

    def load(self, path_prefix):
        """Load model and scaler."""
        self.model = joblib.load(f"{path_prefix}_rf_model.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.is_fitted = True

# ---------------------------------------------------------------------------
# 2. DEEP LEARNING (TensorFlow/Keras)
# ---------------------------------------------------------------------------
class DeepLearningPredictor:
    def __init__(self, model_path=None):
        """
        Initialize DL predictor.
        Requires TensorFlow installed.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        if model_path:
            self.load(model_path)

    def fit(self, X, y):
        """Placeholder: DL model should be trained in notebook and saved as .keras/.h5"""
        raise NotImplementedError(
            "DL model should be trained separately and loaded via .load(). "
            "Use this class only for inference of pre-trained models."
        )

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not loaded. Call .load() first.")
        X_s = self.scaler.transform(X)
        # Assuming binary classification with threshold 0.5
        probs = self.model.predict(X_s, verbose=0).flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not loaded. Call .load() first.")
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s, verbose=0).flatten()

    def save(self, path_prefix):
        """Save Keras model and scaler."""
        self.model.save(f"{path_prefix}_dl_model.keras")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")

    def load(self, path_prefix):
        """Load Keras model and scaler."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(f"{path_prefix}_dl_model.keras")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.is_fitted = True

# ---------------------------------------------------------------------------
# 3. UNIFIED PREDICTOR
# ---------------------------------------------------------------------------
class CentSaverPredictor:
    def __init__(self, rf_path=None, dl_path=None):
        self.rf = RandomForestPredictor(rf_path) if rf_path else None
        self.dl = DeepLearningPredictor(dl_path) if dl_path else None

    def predict(self, X, model_type="rf"):
        if model_type == "rf" and self.rf:
            return self.rf.predict(X)
        elif model_type == "dl" and self.dl:
            return self.dl.predict(X)
        else:
            raise ValueError(f"Model {model_type} not available.")

    def predict_proba(self, X, model_type="rf"):
        if model_type == "rf" and self.rf:
            return self.rf.predict_proba(X)
        elif model_type == "dl" and self.dl:
            return self.dl.predict_proba(X)
        else:
            raise ValueError(f"Model {model_type} not available.")
