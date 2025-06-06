import joblib
import pandas as pd

# Load saved preprocessor and model
preprocessor = joblib.load('preprocessor1.pkl')
model = joblib.load('random_forest_model1.pkl')

# Define input features exactly as in training
feature_columns = [
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 
    'spo2', 'body_temp', 'glucose_level', 'activity_level', 'steps_per_min'
]

# Example user input
user_input_dict = {
    'heart_rate': 79,
    'systolic_bp': 145,
    'diastolic_bp': 78,
    'respiratory_rate': 16,
    'spo2': 93.03885606487918,
    'body_temp': 37.34477941184635,
    'glucose_level': 115.74673708564383,
    'activity_level': 'resting',
    'steps_per_min': 61
}

# Convert input dict to DataFrame with correct column order
user_input_df = pd.DataFrame([user_input_dict])[feature_columns]

# Preprocess user input (only transform, do NOT fit)
processed_input = preprocessor.transform(user_input_df)

# Predict
prediction = model.predict(processed_input)
proba = model.predict_proba(processed_input)
proba_threshold = 0.3 
 # try lower threshold to catch more anomalies
if proba[0][1] > proba_threshold:
    print("Anomaly Detected!")
else:
    print("No Anomaly.")
