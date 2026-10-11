
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "processed_data.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "feedback",
    "error",
    "feedback_change",
    "absolute_error"
]
TARGET = "label"


def main():
    if not os.path.exists(DATA_PATH):
        print("Dataset not found:", DATA_PATH)
        print("Run: python src/extract_features.py")
        return

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("Missing dataset columns:", missing)
        return

    df = df.dropna(subset=required).copy()
    X = df[FEATURES]
    y = df[TARGET].astype(str)

    if y.nunique() < 2:
        print("Dataset must contain at least two label classes.")
        return

    if "run_id" in df.columns and df["run_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(X, y, groups=df["run_id"])
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            ))
        ])
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"\n===== {name.upper()} =====")
        print("Test accuracy:", round(
            accuracy_score(y_test, predictions), 4
        ))
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

        # Refit on all labeled data for the application.
        model.fit(X, y)

        path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump({
            "model": model,
            "features": FEATURES,
            "model_name": name
        }, path)

        print("Saved:", path)


if __name__ == "__main__":
    main()
