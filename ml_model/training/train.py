import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.preprocess import load_and_process_data, normalize
from models.lstm_model import build_model


DATA_PATH = "A:/Knowlegde retention/ml_model/data/your_dataset.csv"
MODEL_PATH = "../saved_models/model.h5"


def main():
    print("📥 Loading data...")
    X, y = load_and_process_data(DATA_PATH)

    print("🔧 Normalizing...")
    X = normalize(X)

    print("🧠 Building model...")
    model = build_model(X.shape[1])

    print("🚀 Training...")
    model.fit(X, y, epochs=20, batch_size=32)

    print("💾 Saving model...")
    os.makedirs("../saved_models", exist_ok=True)
    model.save(MODEL_PATH)

    print("✅ Training complete!")


if __name__ == "__main__":
    main()