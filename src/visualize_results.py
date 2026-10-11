
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_file = BASE_DIR / "data" / "processed" / "processed_data.csv"
plots_dir = BASE_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)

# Graph 1: Feedback readings
plt.figure()
plt.plot(df["timestamp"], df["feedback"], marker="o")
plt.xlabel("Time (seconds)")
plt.ylabel("Feedback Reading")
plt.title("LoopDoctor - Feedback Signal")
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "feedback_signal.png")
plt.close()

# Graph 2: Absolute error
plt.figure()
plt.plot(df["timestamp"], df["error"].abs(), marker="o")
plt.xlabel("Time (seconds)")
plt.ylabel("Absolute Error")
plt.title("LoopDoctor - Control Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "control_error.png")
plt.close()

# Graph 3: Number of samples per class
plt.figure()
df["label"].value_counts().plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("LoopDoctor - Dataset Classes")
plt.tight_layout()
plt.savefig(plots_dir / "class_counts.png")
plt.close()

print("Graphs created successfully!")
print("Saved in:", plots_dir)
