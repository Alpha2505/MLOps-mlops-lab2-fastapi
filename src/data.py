"""Data loading and shared paths for the Wine cultivar classifier."""
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"

# Order must match the column order of sklearn's load_wine().
# (The original "od280/od315_of_diluted_wines" is renamed: "/" isn't valid in a field name.)
FEATURE_ORDER = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]


def load_data():
    """Return features X, labels y, and the class names."""
    wine = load_wine()
    return wine.data, wine.target, [str(n) for n in wine.target_names]


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split so all three cultivars appear in the test set."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
