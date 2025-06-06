import pandas as pd
import joblib

def predict_user_input(user_input_dict, model_path, preprocessor_path):
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input_dict])

    # Preprocess user input
    processed_input = preprocessor.transform(user_input_df)

    # Predict
    prediction = model.predict(processed_input)[0]
    proba = model.predict_proba(processed_input)[0, 1]

    print(f"Prediction: {'Anomaly Detected!' if prediction == 1 else 'No Anomaly.'} (Probability: {proba:.2f})")

# Example usage:
if __name__ == "__main__":
    user_input = {
        'heart_rate': 0,
        'systolic_bp': 150,
        'diastolic_bp': 0,
        'respiratory_rate': 0,
        'spo2': 0,
        'body_temp': 0.3,
        'glucose_level': 0,
        'activity_level': 'intense',
        'steps_per_min': 0
    }

    model_path = r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\src\xgb_model.pkl"
    preprocessor_path = r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\src\xgb_preprocessor.pkl"

    predict_user_input(user_input, model_path, preprocessor_path)
