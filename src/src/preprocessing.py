import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
