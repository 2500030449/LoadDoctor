
import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_file = BASE_DIR / "models" / "feedback_noise_model.pkl"

# Load trained model
saved = joblib.load(model_file)
model = saved["model"]
features = saved["features"]

print("LoopDoctor - Feedback Noise Detector")
print("-----------------------------------")

while True:
    try:
        feedback_text = input(
            "\nEnter feedback reading (or q to quit): "
        )

        if feedback_text.lower() == "q":
            print("Prediction stopped.")
            break

        feedback = float(feedback_text)
        setpoint = float(input("Enter setpoint: "))

        error = setpoint - feedback
        feedback_change = float(input("Enter change from previous feedback: "))
        absolute_error = abs(error)

        # Prepare input in the same feature order used during training
        input_data = pd.DataFrame([{
            "feedback": feedback,
            "error": error,
            "feedback_change": feedback_change,
            "absolute_error": absolute_error
        }])[features]

        # Predict
        prediction = model.predict(input_data)[0]

        print("\nDiagnosis:", prediction)

        if prediction == "feedback_noise":
            print("Advice: Inspect the feedback sensor and its wiring.")
        else:
            print("Advice: No feedback noise detected by this model.")

        print("Note: This is a software prediction, not a safety guarantee.")

    except ValueError:
        print("Please enter valid numbers.")
