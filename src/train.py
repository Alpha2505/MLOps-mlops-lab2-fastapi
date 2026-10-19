"""Train a Random Forest on the Wine dataset and save it with its metadata."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data import FEATURE_ORDER, MODEL_PATH, load_data, split_data


def main():
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save the model together with what the API needs to describe it.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_names": target_names,
            "feature_order": FEATURE_ORDER,
            "test_accuracy": round(float(accuracy), 4),
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
