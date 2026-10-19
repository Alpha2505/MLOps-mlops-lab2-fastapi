import sys
from pathlib import Path

import pytest

# Make src/ importable the same way uvicorn sees it when run from src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from data import MODEL_PATH

    if not MODEL_PATH.exists():
        import train
        train.main()

    from main import app
    return TestClient(app)


@pytest.fixture
def sample():
    """First row of the real dataset, as a request body."""
    from data import FEATURE_ORDER, load_data

    X, _, _ = load_data()
    return dict(zip(FEATURE_ORDER, X[0].tolist()))
