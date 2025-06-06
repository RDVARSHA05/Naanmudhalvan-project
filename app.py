import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load model and reference data (run once)
@st.cache_data
def load_data_and_model():
    model = joblib.load('random_forest_model.pkl')
    ref_df = pd.read_csv(r'C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\synthetic_wearable_health_data.csv')
    ref_df.drop(['timestamp', 'anomaly'], axis=1, inplace=True)
    ref_df['steps_per_min'] = pd.to_numeric(ref_df['steps_per_min'], errors='coerce')
    ref_df.fillna(ref_df.median(numeric_only=True), inplace=True)
    return model, ref_df

# Preprocessing function
def preprocess_user_input(user_input_df, reference_df):
    numeric_features = reference_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = reference_df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    preprocessor.fit(reference_df)
    return preprocessor.transform(user_input_df)

# Load once
model, reference_data = load_data_and_model()

st.title("Wearable Health Data Anomaly Detection")

with st.form("input_form"):
    heart_rate = st.number_input("Heart Rate (bpm)", value=85)
    systolic_bp = st.number_input("Systolic BP (mmHg)", value=130)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", value=82)
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", value=18)
    spo2 = st.number_input("SpO2 (%)", value=96.5, format="%.1f")
    body_temp = st.number_input("Body Temperature (°C)", value=36.9, format="%.1f")
    glucose_level = st.number_input("Glucose Level (mg/dL)", value=100.2, format="%.1f")
    activity_level = st.selectbox("Activity Level", options=['resting', 'moderate', 'intense'])
    steps_per_min = st.number_input("Steps per Minute", value=65)

    submitted = st.form_submit_button("Predict Anomaly")


if submitted:
    user_input_dict = {
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'respiratory_rate': respiratory_rate,
        'spo2': spo2,
        'body_temp': body_temp,
        'glucose_level': glucose_level,
        'activity_level': activity_level,
        'steps_per_min': steps_per_min
    }

    user_input_df = pd.DataFrame([user_input_dict])
    processed_input = preprocess_user_input(user_input_df, reference_data)
    prediction = model.predict(processed_input)

    if prediction[0] == 0:
        st.error("⚠️ Anomaly Detected!")
    else:
        st.success("✅ No Anomaly Detected.")
