import joblib
import pandas as pd
import os

# Get correct path to model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "telecom_model.pkl")

model = joblib.load(model_path)

def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]