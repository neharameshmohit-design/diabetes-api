from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ Load model safely
try:
    model = joblib.load("models/diabetes_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print("❌ Model loading error:", e)

# ✅ Input schema
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    bloodPressure: float
    skinThickness: float
    insulin: float
    bmi: float
    diabetesPedigreeFunction: float
    age: int

# ✅ Home route (TEST THIS IN BROWSER)
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API Running"}

# ✅ Prediction API
@app.post("/predict")
def predict(data: DiabetesInput):

    try:
        input_data = np.array([[
            data.pregnancies,
            data.glucose,
            data.bloodPressure,
            data.skinThickness,
            data.insulin,
            data.bmi,
            data.diabetesPedigreeFunction,
            data.age
        ]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return {
            "prediction": result,
            "risk_score": round(probability * 100, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }