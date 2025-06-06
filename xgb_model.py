import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import xgboost as xgb

def preprocess_data(X_train, X_test, reference_df):
    numeric_features = reference_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = reference_df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # Fit on training data only
    preprocessor.fit(X_train)

    # Transform both train and test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor

def train_and_evaluate(df):
    # Prepare features and label
    y = df['anomaly']
    X = df.drop(['anomaly', 'timestamp'], axis=1, errors='ignore')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test, X)

    # Train XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_processed, y_train)

    # Predict
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Feature importance
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Save model and preprocessor
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(preprocessor, 'xgb_preprocessor.pkl')

    print("Model and preprocessor saved.")
    return model, preprocessor

def predict_user_input(user_input_dict, model, preprocessor):
    # Convert input to DataFrame
    user_input_df = pd.DataFrame([user_input_dict])

    # Preprocess user input
    processed_input = preprocessor.transform(user_input_df)

    # Predict
    prediction = model.predict(processed_input)[0]
    proba = model.predict_proba(processed_input)[0, 1]

    print(f"Prediction: {'Anomaly Detected!' if prediction == 1 else 'No Anomaly.'} (Probability: {proba:.2f})")

if __name__ == "__main__":
    # Load and prepare your dataset
    df = pd.read_csv(r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\synthetic_wearable_health_data.csv")

    # Clean data
    if 'timestamp' in df.columns:
        df.drop('timestamp', axis=1, inplace=True)
    df['steps_per_min'] = pd.to_numeric(df['steps_per_min'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Train and evaluate
    model, preprocessor = train_and_evaluate(df)

    # Example user input to test prediction
    user_input = {
        'heart_rate': 79,
        'systolic_bp': 145,
        'diastolic_bp': 78,
        'respiratory_rate': 16,
        'spo2': 93.0,
        'body_temp': 37.3,
        'glucose_level': 115,
        'activity_level': 'resting',
        'steps_per_min': 61
    }

    predict_user_input(user_input, model, preprocessor)
