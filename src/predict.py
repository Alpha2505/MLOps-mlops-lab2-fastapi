"""Load the saved model once and turn feature rows into predictions."""
from functools import lru_cache

import joblib
import numpy as np

from data import MODEL_PATH


@lru_cache(maxsize=1)
def load_bundle():
    """Load model + metadata from disk (cached after the first call)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def predict(rows):
    """rows: list of feature lists in FEATURE_ORDER. Returns one dict per row."""
    bundle = load_bundle()
    probs = bundle["model"].predict_proba(np.asarray(rows, dtype=float))
    best = probs.argmax(axis=1)
    return [
        {
            "class_id": int(i),
            "class_name": bundle["target_names"][i],
            "confidence": round(float(p[i]), 4),
        }
        for i, p in zip(best, probs)
    ]
