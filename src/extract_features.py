
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "raw" / "sample_readings.csv"
output_file = BASE_DIR / "data" / "processed" / "processed_data.csv"

# Create sample CSV automatically if it is missing
if not input_file.exists():
    input_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": [i / 100 for i in range(20)],
        "feedback": [
            100, 101, 99, 100, 102, 98, 100, 101, 99, 100,
            120, 80, 125, 75, 130, 70, 122, 78, 128, 72
        ],
        "setpoint": [100] * 20,
        "label": ["normal"] * 10 + ["feedback_noise"] * 10
    }

    sample_df = pd.DataFrame(data)
    sample_df["error"] = sample_df["setpoint"] - sample_df["feedback"]
    sample_df.to_csv(input_file, index=False)

    print("Sample CSV created:", input_file)

# Read the CSV
df = pd.read_csv(input_file)

# Calculate features
df["error"] = df["setpoint"] - df["feedback"]
df["feedback_change"] = df["feedback"].diff().fillna(0)
df["absolute_error"] = df["error"].abs()

# Create processed folder and save results
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("Feature extraction completed!")
print("Processed file:", output_file)
print("Total rows:", len(df))
print(df.head())