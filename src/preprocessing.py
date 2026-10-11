
import pandas as pd


def preprocess_data(input_file, output_file):
    # Read the raw CSV file
    df = pd.read_csv(input_file)

    # Remove completely duplicated rows
    df = df.drop_duplicates()

    # Keep rows in timestamp order, if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Convert feedback readings to numbers
    if "feedback" in df.columns:
        df["feedback"] = pd.to_numeric(
            df["feedback"], errors="coerce"
        )

        # Remove rows where feedback is missing
        df = df.dropna(subset=["feedback"])

    # Save the cleaned data
    df.to_csv(output_file, index=False)

    print("Preprocessing completed!")
    print("Cleaned data saved to:", output_file)
    print("Rows remaining:", len(df))


if __name__ == "__main__":
    preprocess_data(
        "data/raw/readings.csv",
        "data/processed/cleaned_data.csv"
    )
