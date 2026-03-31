import numpy as np
import os
from tensorflow.keras.models import load_model

# Get current file directory (backend/model/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct path to model
MODEL_PATH = os.path.join(BASE_DIR, "../../ml_model/saved_models/model.h5")

# Load trained model
model = load_model(MODEL_PATH, compile=False)


def model_predict(features):
    prediction = model.predict(features)[0][0]
    return float(prediction)