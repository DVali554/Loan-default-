import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Find CSV file
csv_file = 'Loan_default.csv' if os.path.exists('Loan_default.csv') else 'data/Loan_default.csv'
print(f"Loading data from {csv_file}...")
df = pd.read_csv(csv_file)

# Drop LoanID if present
if 'LoanID' in df.columns:
    df = df.drop(['LoanID'], axis=1)

# Encode categorical features
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']
le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Separate features and target
X = df.drop('Default', axis=1)
y = df['Default']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
print("Applying SMOTE oversampling...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train model
print("Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save artifacts
joblib.dump(rf_model, 'credit_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')

print("Success: Model and artifacts saved successfully!")
