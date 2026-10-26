"""
ReAct tool: detect_seizure.
"""
import os
import time
import numpy as np
import joblib

from src.tools.features import _extract_features

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "seizure_rf_model.pkl"
)

_model = None


def _get_model():
    """Lazy-load the classifier so importing this module never requires a trained model."""
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {_MODEL_PATH}. Run train_model.py first."
            )
        _model = joblib.load(_MODEL_PATH)
    return _model


def detect_seizure(eeg_window: list, fs: int = 256) -> dict:
    """
    Classify a 5-second, 6-channel EEG window.

    Returns:
        {
            "probability": float,
            "label": "seizure" | "preictal" | "interictal",
            "confidence": "high" | "medium" | "low",
            "latency_ms": float,
        }
    """
    t0 = time.time()
    window = np.asarray(eeg_window, dtype=np.float32)
    feats = _extract_features(window, fs=fs)
    prob = float(_get_model().predict_proba(feats)[0][1])


    #if prob >= 0.75:
        #label, conf = "seizure", "high"
    #elif 0.45 <= prob < 0.75:
        #label, conf = "preictal", "medium"
    #else:
        #label, conf = "interictal", "low"

    if prob >= 0.50 :
        label,conf ="seizure" , "high"
    elif prob >=30 :
        label , conf = "preictal" , "medium"
    else:
        label , conf ="interictal" , "low"

    return {
        "probability": round(prob, 4),
        "label": label,
        "confidence": conf,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

    
    
    
    
    