from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

import boto3


# Load model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

S3_BUCKET = "shreya-aws-churn-model-2026"
S3_KEY = "churn-model/churn_model.pkl"

LOCAL_MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)


# Input schema
class Customer(BaseModel):
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


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):

    # Convert request to DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Prediction
    prediction = model.predict(data)[0]

    # Probability
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": "Likely to churn" if prediction == 1 else "Likely to stay",
        "churn_probability": round(float(probability), 4)
    }

def download_model_from_s3():
    if not LOCAL_MODEL_PATH.exists():
        LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        s3.download_file(
            S3_BUCKET,
            S3_KEY,
            str(LOCAL_MODEL_PATH)
        )

download_model_from_s3()
model = joblib.load(LOCAL_MODEL_PATH)