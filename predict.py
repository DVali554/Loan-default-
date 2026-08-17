import joblib
import pandas as pd
import numpy as np

# Load saved artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Sample borrower data
sample_data = pd.DataFrame([{
    'Age': 35,
    'Income': 75000,
    'LoanAmount': 20000,
    'CreditScore': 680,
    'MonthsEmployed': 48,
    'NumCreditLines': 3,
    'InterestRate': 12.5,
    'LoanTerm': 36,
    'DTIRatio': 0.35,
    'Education': "Bachelor's",
    'EmploymentType': 'Full-time',
    'MaritalStatus': 'Married',
    'HasMortgage': 'Yes',
    'HasDependents': 'No',
    'LoanPurpose': 'Home',
    'HasCoSigner': 'No'
}])

# Encode categorical columns using saved encoders
for col, le in label_encoders.items():
    if col in sample_data.columns:
        sample_data[col] = le.transform(sample_data[col])

# Scale numeric values and predict
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)[0][1]

print("=" * 40)
print(f"Default Prediction : {'Default' if prediction[0] == 1 else 'No Default'}")
print(f"Default Probability: {probability:.2%}")
print("=" * 40)
