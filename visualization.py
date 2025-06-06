import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def evaluate_and_visualize_model(X_raw, y):
    # Identify numeric and categorical columns
    numeric_features = X_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_raw.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # Preprocess the data
    X_processed = preprocessor.fit_transform(X_raw)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Feature Importance (Note: One-hot encoded names are not preserved)
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importances")
    plt.xlabel("Feature Index (post-encoding)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    # Save model and preprocessor
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    return model, preprocessor

# ---------- Data Loading & Preprocessing ----------
df = pd.read_csv(r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\synthetic_wearable_health_data.csv")

# Clean "steps_per_min" which may have non-numeric values like "0m"
df['steps_per_min'] = pd.to_numeric(df['steps_per_min'], errors='coerce')

# Drop the timestamp if present
if 'timestamp' in df.columns:
    df.drop('timestamp', axis=1, inplace=True)

# Fill missing numeric values
df.fillna(df.median(numeric_only=True), inplace=True)

# Separate features and label
y = df['anomaly']
X = df.drop('anomaly', axis=1)

# Run evaluation
model, preprocessor = evaluate_and_visualize_model(X, y)
