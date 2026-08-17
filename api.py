import io
import os
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Credit Risk Assessment & Loan Default Prediction API",
    description="Production-grade API for scoring credit risk and predicting loan default probabilities.",
    version="1.0.0"
)

# 1. Load ML Pipeline Artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

DB_PATH = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
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

def log_prediction(age, income, loan_amount, credit_score, dti_ratio, prediction, prob):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_logs 
            (timestamp, age, income, loan_amount, credit_score, dti_ratio, prediction, default_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            age, income, loan_amount, credit_score, dti_ratio, prediction, prob
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to DB: {e}")

class LoanApplicant(BaseModel):
    Age: int = Field(..., ge=18, le=100, example=35)
    Income: float = Field(..., ge=1000, example=75000)
    LoanAmount: float = Field(..., ge=500, example=20000)
    CreditScore: int = Field(..., ge=300, le=850, example=680)
    MonthsEmployed: int = Field(..., ge=0, example=48)
    NumCreditLines: int = Field(..., ge=0, example=3)
    InterestRate: float = Field(..., ge=0.0, example=12.5)
    LoanTerm: int = Field(..., example=36)
    DTIRatio: float = Field(..., ge=0.0, le=1.0, example=0.35)
    Education: str = Field(..., example="Bachelor's")
    EmploymentType: str = Field(..., example="Full-time")
    MaritalStatus: str = Field(..., example="Married")
    HasMortgage: str = Field(..., example="Yes")
    HasDependents: str = Field(..., example="No")
    LoanPurpose: str = Field(..., example="Home")
    HasCoSigner: str = Field(..., example="No")

@app.get("/")
def read_root():
    return {"message": "Loan Default Prediction API is live!"}

@app.post("/predict")
def predict_loan_default(applicant: LoanApplicant):
    try:
        df = pd.DataFrame([applicant.model_dump()])

        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = float(model.predict_proba(scaled)[0][1])

        prediction_label = "Default" if pred == 1 else "No Default"

        log_prediction(
            applicant.Age, applicant.Income, applicant.LoanAmount,
            applicant.CreditScore, applicant.DTIRatio,
            prediction_label, round(prob, 4)
        )

        return {
            "prediction": prediction_label,
            "default_probability": round(prob, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        cols_to_drop = [c for c in ['LoanID', 'Default', 'Prediction', 'prediction', 'default_probability', 'Default_Probability'] if c in df.columns]
        features_df = df.drop(columns=cols_to_drop)

        for col, le in label_encoders.items():
            if col in features_df.columns:
                features_df[col] = le.transform(features_df[col].astype(str))

        scaled = scaler.transform(features_df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        output_df = df.copy()
        output_df['Prediction'] = np.where(preds == 1, 'Default', 'No Default')
        output_df['Default_Probability'] = np.round(probs, 4)

        stream = io.StringIO()
        output_df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=scored_batch_predictions.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")

@app.get("/metrics")
def get_metrics():
    if not os.path.exists(DB_PATH):
        return {"total_predictions": 0, "default_count": 0, "default_rate": 0.0}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN prediction = 'Default' THEN 1 ELSE 0 END), AVG(credit_score), AVG(loan_amount) FROM prediction_logs")
    row = cursor.fetchone()
    conn.close()

    total = row[0] or 0
    defaults = row[1] or 0
    default_rate = round((defaults / total), 4) if total > 0 else 0.0
    avg_score = round(row[2], 1) if row[2] else 0.0
    avg_loan = round(row[3], 2) if row[3] else 0.0

    return {
        "total_predictions": total,
        "default_count": defaults,
        "default_rate": default_rate,
        "average_credit_score": avg_score,
        "average_loan_amount": avg_loan
    }
