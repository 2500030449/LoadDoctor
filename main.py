
import os


def main():
    print("LOOPDOCTOR - PROJECT CHECK")
    print("-" * 35)

    folders = [
        "data/raw",
        "data/processed",
        "models",
        "plots",
        "firmware",
        "src"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Checked folder: {folder}")

    print("\nProject folders are ready!")
    print("Next: verify real sensor data before training.")


if __name__ == "__main__":
    main()
