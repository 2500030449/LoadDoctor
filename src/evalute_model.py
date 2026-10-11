
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
data_file = BASE_DIR / "data" / "processed" / "processed_data.csv"
model_file = BASE_DIR / "models" / "feedback_noise_model.pkl"
plot_file = BASE_DIR / "plots" / "confusion_matrix.png"

# Load data and trained model
df = pd.read_csv(data_file)
saved = joblib.load(model_file)

model = saved["model"]
features = saved["features"]

# Prepare test data
X = df[features]
y = df["label"]

# Predict labels
predictions = model.predict(X)

# Print evaluation results
print("Model evaluation completed!")
print("Accuracy:", accuracy_score(y, predictions))
print("\nClassification Report:")
print(classification_report(y, predictions, zero_division=0))

# Create confusion matrix
labels = ["normal", "feedback_noise"]
matrix = confusion_matrix(y, predictions, labels=labels)

print("Confusion Matrix:")
print(matrix)

# Save confusion matrix graph
plot_file.parent.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)
display.plot()
plt.title("LoopDoctor - Feedback Noise Detection")
plt.tight_layout()
plt.savefig(plot_file)
plt.close()

print("\nGraph saved at:", plot_file)