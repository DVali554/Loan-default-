from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Loan Default Prediction API is live!"}

def test_valid_prediction():
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
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0

def test_invalid_payload():
    # Missing required fields
    response = client.post("/predict", json={"Age": 30})
    assert response.status_code == 422
