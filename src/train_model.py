
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
data_file = BASE_DIR / "data" / "processed" / "processed_data.csv"
model_file = BASE_DIR / "models" / "feedback_noise_model.pkl"

# Load processed data
df = pd.read_csv(data_file)

# Features used for prediction
features = [
    "feedback",
    "error",
    "feedback_change",
    "absolute_error"
]

X = df[features]
y = df["label"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
model_file.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    model_file
)

print("\nModel saved at:", model_file)
