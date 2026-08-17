import os
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# 1. Pipeline Artifact Verification
def test_artifacts_exist_and_loadable():
    assert os.path.exists("credit_risk_model.pkl"), "Model file missing"
    assert os.path.exists("scaler.pkl"), "Scaler file missing"
    assert os.path.exists("label_encoders.pkl"), "Encoders file missing"

    model = joblib.load("credit_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")

    assert hasattr(model, "predict"), "Loaded model does not have a predict method"
    assert hasattr(scaler, "transform"), "Loaded scaler does not have a transform method"
    assert isinstance(encoders, dict), "Encoders should be stored in a dictionary"

# 2. API Root Health Check
def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Loan Default Prediction API is live!"}

# 3. Single Prediction Valid Payload
def test_predict_single_applicant():
    payload = {
        "Age": 35,
        "Income": 75000,
        "LoanAmount": 20000,
        "CreditScore": 680,
        "MonthsEmployed": 48,
        "NumCreditLines": 3,
        "InterestRate": 12.5,
        "LoanTerm": 36,
        "DTIRatio": 0.35,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "No",
        "LoanPurpose": "Home",
        "HasCoSigner": "No"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Default", "No Default"]
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0

# 4. Single Prediction Schema Validation (Invalid Payload)
def test_predict_invalid_payload():
    invalid_payload = {
        "Age": "not_a_number",
        "Income": 75000
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity

# 5. Metrics & SQLite Audit Logging Endpoint
def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "default_count" in data
    assert "default_rate" in data

# 6. Batch CSV Prediction Endpoint
def test_predict_batch_csv():
    csv_data = """Age,Income,LoanAmount,CreditScore,MonthsEmployed,NumCreditLines,InterestRate,LoanTerm,DTIRatio,Education,EmploymentType,MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner
35,75000,20000,680,48,3,12.5,36,0.35,Bachelor's,Full-time,Married,Yes,No,Home,No
22,24000,45000,510,4,8,24.0,60,0.65,High School,Unemployed,Single,No,Yes,Other,No
"""
    files = {"file": ("test.csv", csv_data, "text/csv")}
    response = client.post("/predict-batch", files=files)
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert "Prediction" in response.text
    assert "Default_Probability" in response.text
