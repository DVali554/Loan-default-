import streamlit as st
import pandas as pd
import joblib

# Load serialized artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Enter applicant financial and demographic details to evaluate default risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Annual Income ($)", 1000, 1000000, 75000)
    loan_amount = st.number_input("Loan Amount ($)", 500, 500000, 20000)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    months_employed = st.number_input("Months Employed", 0, 600, 48)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 30, 3)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.5)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)

with col2:
    dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.35)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
    cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])

if st.button("Predict Default Risk", type="primary"):
    data = pd.DataFrame([{
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education,
        'EmploymentType': employment, 'MaritalStatus': marital_status,
        'HasMortgage': mortgage, 'HasDependents': dependents,
        'LoanPurpose': purpose, 'HasCoSigner': cosigner
    }])

    for col, le in label_encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ **High Risk: Likely to Default** (Estimated Default Probability: **{prob:.1%}**)")
    else:
        st.success(f"✅ **Low Risk: No Default Expected** (Estimated Default Probability: **{prob:.1%}**)")
