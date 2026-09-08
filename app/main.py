from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)

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