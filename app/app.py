from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Initialize FastAPI

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn using an XGBoost model.",
    version="1.0.0"
)

# Load Trained Model

MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

# Request Schema

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Home Endpoint

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running!",
        "model": "XGBoost",
        "status": "Success"
    }


# Prediction Endpoint  

@app.post("/predict")
def predict(customer: CustomerData):

    # Convert request to DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Prediction
    prediction = model.predict(customer_df)[0]

    # Probability
    probability = model.predict_proba(customer_df)[0]

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability[1]), 4),
        "no_churn_probability": round(float(probability[0]), 4)
    }