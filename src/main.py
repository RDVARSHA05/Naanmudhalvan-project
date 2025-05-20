main.py
from src.preprocessing import load_and_preprocess_data
from src.model import train_and_save_model

if __name__ == "__main__":
    X, y = load_and_preprocess_data("data/heart_disease.csv")
    train_and_save_model(X, y)
