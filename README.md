# FastAPI Lab 1: Wine Cultivar Classifier API

Serves a scikit-learn Random Forest trained on the UCI Wine dataset as a REST API using FastAPI and uvicorn.

## How this differs from the course lab

<!-- Write this section in your own words. Points you can cover: -->
| | Course lab | This repo |
|---|---|---|
| Dataset | Iris (4 features) | Wine (13 chemical features, 3 cultivars) |
| Model | Decision Tree | Random Forest (200 trees), saved with metadata |
| Endpoints | `/predict` | `/health`, `/model-info`, `/predict`, `/predict/batch` |
| Response | class id | class id, class name, confidence |
| Validation | types only | non-negative constraints, batch size limits (400), 503 if model missing |
| Tests | none | pytest suite using FastAPI `TestClient` |

## Setup

```bash
python -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate      # Windows: fastapi_lab1_env\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
cd src
python train.py
```

## Run the API

```bash
uvicorn main:app --reload      # from inside src/
```

Open http://127.0.0.1:8000/docs to try the endpoints.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Model type, classes, features, test accuracy |
| POST | `/predict` | Predict one wine sample |
| POST | `/predict/batch` | Predict up to 50 samples at once |

Example request body for `/predict`:

```json
{
  "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
  "magnesium": 127, "total_phenols": 2.8, "flavanoids": 3.06,
  "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
  "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065
}
```

## Tests

```bash
pytest -v      # from the repo root
```

## Screenshots

<!-- Add screenshots of /docs and a successful /predict response to assets/ and link them here. -->
