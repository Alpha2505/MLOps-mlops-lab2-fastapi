"""FastAPI app serving the Wine cultivar classifier."""
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import FEATURE_ORDER
from predict import load_bundle, predict

MAX_BATCH = 50

app = FastAPI(
    title="Wine Cultivar Classifier",
    description="Predicts which of three cultivars a wine came from, based on 13 chemical measurements.",
    version="1.0.0",
)


class WineSample(BaseModel):
    # ge=0: none of these chemical measurements can be negative.
    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)

    # Pre-fills "Try it out" in /docs with a real sample from the dataset.
    model_config = {
        "json_schema_extra": {
            "example": {
                "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
                "alcalinity_of_ash": 15.6, "magnesium": 127, "total_phenols": 2.8,
                "flavanoids": 3.06, "nonflavanoid_phenols": 0.28,
                "proanthocyanins": 2.29, "color_intensity": 5.64, "hue": 1.04,
                "od280_od315_of_diluted_wines": 3.92, "proline": 1065,
            }
        }
    }


class WinePrediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class BatchRequest(BaseModel):
    samples: List[WineSample]


class BatchResponse(BaseModel):
    count: int
    predictions: List[WinePrediction]


def to_row(sample: WineSample):
    return [getattr(sample, name) for name in FEATURE_ORDER]


def require_model():
    """Return 503 instead of crashing if the model hasn't been trained yet."""
    try:
        return load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    bundle = require_model()
    return {
        "model_type": type(bundle["model"]).__name__,
        "classes": bundle["target_names"],
        "features": bundle["feature_order"],
        "test_accuracy": bundle["test_accuracy"],
    }


@app.post("/predict", response_model=WinePrediction)
def predict_one(sample: WineSample):
    require_model()
    return predict([to_row(sample)])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if not request.samples:
        raise HTTPException(status_code=400, detail="samples must not be empty")
    if len(request.samples) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_BATCH} samples per request (got {len(request.samples)})",
        )
    require_model()
    preds = predict([to_row(s) for s in request.samples])
    return {"count": len(preds), "predictions": preds}
