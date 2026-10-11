# LoopDoctor – Automatic Control-Pathology Classifier

## 1. Project Overview

LoopDoctor is a machine-learning-based prototype designed to identify abnormal feedback signals in a control system.

It analyzes feedback readings, calculates control error, and classifies the signal as normal or feedback noise. It also provides suggestions for inspection.

**Note:** The current software prototype uses simulated data. Live hardware integration is not yet verified.

## 2. Objectives

- Monitor feedback signals.
- Calculate the difference between the setpoint and feedback.
- Extract features from sensor readings.
- Classify feedback conditions using machine learning.
- Display predictions and visualizations.
- Recommend inspection actions.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 4. Project Workflow

1. Generate or collect feedback readings.
2. Store the readings in a CSV file.
3. Extract features such as error, absolute error, and feedback change.
4. Train a Random Forest classifier.
5. Evaluate the model and generate a confusion matrix.
6. Visualize feedback signals and control errors.
7. Display a diagnosis and inspection suggestion through the dashboard.

## 5. Machine Learning

**Algorithm:** Random Forest Classifier

**Input features:**
- Feedback
- Control error
- Feedback change
- Absolute error

**Target classes:**
- Normal
- Feedback noise

The model learns patterns from labelled sample data to predict the class of a new reading.

## 6. How to Run

Run these commands from the project root directory:

```bash
python src/extract_features.py
python src/train_model.py
python src/evaluate_model.py
python src/visualize_results.py
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the dashboard.

## 7. Limitations and Future Work

- Current results use simulated sample data.
- Training and evaluation must be improved using independent experimental runs.
- Live STM32 telemetry integration remains to be verified.
- Hardware testing requires the approved controller, motor driver, encoder, current sensing, and safety equipment.
- Machine-learning predictions must not directly control PWM or PID parameters.

## 8. Conclusion

LoopDoctor demonstrates a software pipeline for detecting feedback-signal abnormalities using feature extraction, machine learning, visualizations, and inspection recommendations. Future work will validate the model on independent experimental data and integrate approved hardware safely.