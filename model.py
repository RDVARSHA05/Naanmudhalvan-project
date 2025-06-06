# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# def train_and_save_model(X, y, model_path='random_forest_model.pkl'):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     print("\nModel Performance:\n")
#     print(classification_report(y_test, y_pred))

#     joblib.dump(model, model_path)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load and preprocess data
df = pd.read_csv(r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\synthetic_wearable_health_data.csv")

# Clean steps_per_min
df['steps_per_min'] = pd.to_numeric(df['steps_per_min'], errors='coerce')

# Drop unused columns
if 'timestamp' in df.columns:
    df.drop('timestamp', axis=1, inplace=True)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Split features and target
y = df['anomaly']
X = df.drop('anomaly', axis=1)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Fit and transform training data
X_processed = preprocessor.fit_transform(X)

# Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the preprocessor and model
joblib.dump(preprocessor, 'preprocessor1.pkl')
joblib.dump(model, 'random_forest_model1.pkl')

print("Training complete and models saved.")
