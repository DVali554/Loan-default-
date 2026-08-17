import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Loan Default Predictor & Explainability", layout="wide")

@st.cache_resource
def load_pipeline():
    if os.path.exists('credit_risk_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('label_encoders.pkl'):
        model = joblib.load('credit_risk_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, scaler, encoders

    csv_path = 'data/Loan_default.csv' if os.path.exists('data/Loan_default.csv') else 'Loan_default.csv'
    df = pd.read_csv(csv_path)
    if 'LoanID' in df.columns:
        df = df.drop(['LoanID'], axis=1)

    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                        'HasDependents', 'LoanPurpose', 'HasCoSigner']
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df.drop('Default', axis=1)
    y = df['Default']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, _, y_train, _ = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, scaler, encoders

model, scaler, label_encoders = load_pipeline()

# Create SHAP TreeExplainer
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.title("🏦 Loan Default Risk Predictor & Explainability")
st.markdown("Predict borrower default risk and visualize feature contributions with SHAP.")

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

feature_names = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
                 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
                 'LoanPurpose', 'HasCoSigner']

if st.button("Predict & Explain Risk", type="primary"):
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

    st.subheader("🔍 Decision Factors (SHAP Explanation)")
    st.write("Factors pushing the prediction towards default appear in red, while factors reducing risk appear in blue.")

    shap_explanation = explainer(scaled)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_explanation[0, :, 1], show=False)
    st.pyplot(fig)
    plt.close()
