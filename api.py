import io
import sqlite3
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Prediction API")

# Initialize database table for audit logs
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            income REAL,
            loan_amount REAL,
            credit_score INTEGER,
            dti_ratio REAL,
            prediction TEXT,
            default_probability REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

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
    pred_label = "Default" if pred == 1 else "No Default"

    # Save transaction to SQLite
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_logs (timestamp, age, income, loan_amount, credit_score, dti_ratio, prediction, default_probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        applicant.Age,
        applicant.Income,
        applicant.LoanAmount,
        applicant.CreditScore,
        applicant.DTIRatio,
        pred_label,
        round(prob, 4)
    ))
    conn.commit()
    conn.close()

    return {
        "prediction": pred_label,
        "default_probability": round(prob, 4)
    }

@app.get("/metrics")
def get_metrics():
    conn = sqlite3.connect("predictions.db")
    df_logs = pd.read_sql_query("SELECT * FROM prediction_logs", conn)
    conn.close()

    if df_logs.empty:
        return {"total_predictions": 0, "default_rate": 0.0}

    total = len(df_logs)
    defaults = len(df_logs[df_logs['prediction'] == "Default"])
    return {
        "total_predictions": total,
        "default_count": defaults,
        "default_rate": round(defaults / total, 4),
        "average_credit_score": round(df_logs['credit_score'].mean(), 1),
        "average_loan_amount": round(df_logs['loan_amount'].mean(), 2)
    }
