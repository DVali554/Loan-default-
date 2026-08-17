from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

class Applicant(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is live!"}

@app.post("/predict")
def predict_default(applicant: Applicant):
    data = pd.DataFrame([applicant.model_dump()])
    for col, le in label_encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])
    
    scaled = scaler.transform(data)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])

    return {
        "prediction": "Default" if pred == 1 else "No Default",
        "default_probability": round(prob, 4)
    }
