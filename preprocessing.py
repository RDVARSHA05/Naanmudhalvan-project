import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Remove rows with invalid numeric data (like "0m" in steps_per_min)
    df['steps_per_min'] = pd.to_numeric(df['steps_per_min'], errors='coerce')

    # Drop the timestamp column
    if 'timestamp' in df.columns:
        df.drop('timestamp', axis=1, inplace=True)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Separate target variable
    y = df['anomaly']
    X = df.drop('anomaly', axis=1)

    # Identify column types
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y
