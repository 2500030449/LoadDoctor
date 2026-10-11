
import csv
import os
from datetime import datetime

import serial


def collect_data():
    port = input("Enter STM32 serial port (example: COM3): ")
    baud_rate = 115200

    os.makedirs("data/raw", exist_ok=True)

    filename = datetime.now().strftime(
        "data/raw/telemetry_%Y%m%d_%H%M%S.csv"
    )

    try:
        with serial.Serial(port, baud_rate, timeout=2) as board:
            print("Connected to:", port)
            print("Press Ctrl+C to stop collecting data.")

            with open(filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "raw_reading"])

                while True:
                    line = board.readline().decode(
                        "utf-8", errors="replace"
                    ).strip()

                    if line:
                        timestamp = datetime.now().isoformat()
                        writer.writerow([timestamp, line])
                        file.flush()
                        print(timestamp, line)

    except KeyboardInterrupt:
        print("\nData collection stopped.")
        print("Saved file:", filename)

    except serial.SerialException as error:
        print("Serial connection error:", error)


if __name__ == "__main__":
    collect_data()
