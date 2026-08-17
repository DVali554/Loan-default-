import io
import os
import sqlite3
from datetime import datetime
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

st.set_page_config(page_title="Credit Risk AI Platform", layout="wide")

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

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.title("🏦 Credit Risk Assessment & Monitoring System")

tab1, tab2, tab3 = st.tabs([
    "🔮 Single Applicant Prediction & SHAP", 
    "📁 Batch CSV Scoring", 
    "📊 Audit Logs & Analytics"
])

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tab1:
    st.markdown("Enter applicant details to generate predictions and feature explanations.")
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

    if st.button("Evaluate Credit Risk", type="primary"):
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
            st.error(f"⚠️ **High Risk: Default Predicted** (Estimated Default Probability: **{prob:.1%}**)")
        else:
            st.success(f"✅ **Low Risk: No Default Predicted** (Estimated Default Probability: **{prob:.1%}**)")

        st.subheader("🔍 SHAP Explanation")
        shap_explanation = explainer(scaled)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation[0, :, 1], show=False)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        st.download_button(
            label="📥 Download SHAP Explanation (PNG)",
            data=buf,
            file_name="shap_explanation_applicant.png",
            mime="image/png"
        )

# ----------------- TAB 2: BATCH CSV SCORING -----------------
with tab2:
    st.subheader("📁 Bulk Applicant Scoring")
    st.markdown("Upload a CSV file containing applicant records. The model will score each row, append default predictions and probabilities, and generate a downloadable report.")

    # Sample template generator
    sample_template_df = pd.DataFrame([
        {
            "Age": 35, "Income": 75000, "LoanAmount": 20000, "CreditScore": 680,
            "MonthsEmployed": 48, "NumCreditLines": 3, "InterestRate": 12.5,
            "LoanTerm": 36, "DTIRatio": 0.35, "Education": "Bachelor's",
            "EmploymentType": "Full-time", "MaritalStatus": "Married",
            "HasMortgage": "Yes", "HasDependents": "No", "LoanPurpose": "Home",
            "HasCoSigner": "No"
        },
        {
            "Age": 24, "Income": 32000, "LoanAmount": 15000, "CreditScore": 580,
            "MonthsEmployed": 12, "NumCreditLines": 5, "InterestRate": 18.0,
            "LoanTerm": 48, "DTIRatio": 0.48, "Education": "High School",
            "EmploymentType": "Part-time", "MaritalStatus": "Single",
            "HasMortgage": "No", "HasDependents": "Yes", "LoanPurpose": "Auto",
            "HasCoSigner": "No"
        },
        {
            "Age": 45, "Income": 120000, "LoanAmount": 40000, "CreditScore": 760,
            "MonthsEmployed": 96, "NumCreditLines": 4, "InterestRate": 7.5,
            "LoanTerm": 60, "DTIRatio": 0.20, "Education": "Master's",
            "EmploymentType": "Full-time", "MaritalStatus": "Married",
            "HasMortgage": "Yes", "HasDependents": "Yes", "LoanPurpose": "Business",
            "HasCoSigner": "Yes"
        }
    ])

    template_buffer = io.BytesIO()
    sample_template_df.to_csv(template_buffer, index=False)
    template_buffer.seek(0)

    # Template download action
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=template_buffer.getvalue(),
        file_name="loan_applicants_template.csv",
        mime="text/csv",
        help="Click to download a formatted CSV file with required headers and example rows."
    )

    st.divider()

    uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"**Loaded {len(raw_df)} applicant records:**")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if st.button("Score Batch Applicants", type="primary"):
            scoring_df = raw_df.copy()
            
            # Drop target or identifier columns if present
            cols_to_drop = [c for c in ['LoanID', 'Default', 'Prediction', 'prediction', 'default_probability', 'Default_Probability'] if c in scoring_df.columns]
            features_df = scoring_df.drop(columns=cols_to_drop)

            # Apply label encodings
            for col, le in label_encoders.items():
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype(str)
                    features_df[col] = le.transform(features_df[col])

            scaled_batch = scaler.transform(features_df)
            preds = model.predict(scaled_batch)
            probs = model.predict_proba(scaled_batch)[:, 1]

            # Append results to original dataset
            output_df = raw_df.copy()
            output_df['Prediction'] = np.where(preds == 1, 'Default', 'No Default')
            output_df['Default_Probability'] = np.round(probs, 4)

            # Summary Metrics
            total_scored = len(output_df)
            total_defaults = int(np.sum(preds == 1))
            default_pct = (total_defaults / total_scored) * 100

            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Scored", f"{total_scored:,}")
            b2.metric("Predicted Defaults", f"{total_defaults:,}")
            b3.metric("Projected Default Rate", f"{default_pct:.2f}%")

            # Display Scored Table
            st.subheader("Scored Results Preview")
            st.dataframe(output_df, use_container_width=True)

            # CSV Download Button
            csv_buffer = io.BytesIO()
            output_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="📥 Download Scored CSV Report",
                data=csv_buffer.getvalue(),
                file_name=f"scored_loans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ----------------- TAB 3: AUDIT LOGS & ANALYTICS -----------------
with tab3:
    st.subheader("📈 Live Database Inference Metrics")
    if os.path.exists("predictions.db"):
        conn = sqlite3.connect("predictions.db")
        df_logs = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
        conn.close()

        if not df_logs.empty:
            m1, m2, m3, m4 = st.columns(4)
            total = len(df_logs)
            defaults = len(df_logs[df_logs['prediction'] == "Default"])
            
            m1.metric("Total Inferences", total)
            m2.metric("Default Rate", f"{(defaults / total):.1%}")
            m3.metric("Avg Credit Score", f"{df_logs['credit_score'].mean():.0f}")
            m4.metric("Avg Loan Amount", f"${df_logs['loan_amount'].mean():,.0f}")

            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No prediction logs recorded yet. Send requests via the API to populate logs.")
    else:
        st.info("Database file `predictions.db` not found. Start the FastAPI service to begin tracking.")
