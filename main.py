from preprocessing import load_and_preprocess_data
from model import train_and_save_model

if __name__ == "__main__":
    X, y = load_and_preprocess_data(r"C:\Users\rdvar\OneDrive\Desktop\NM-Project\Naanmudhalvan-project\synthetic_wearable_health_data.csv")
    train_and_save_model(X, y)
