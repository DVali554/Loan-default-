import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

# Load artifacts and test dataset
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

df = pd.read_csv('Loan_default.csv')
if 'LoanID' in df.columns:
    df = df.drop(['LoanID'], axis=1)

for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

X = scaler.transform(df.drop('Default', axis=1))
y = df['Default'].values

# Split holdout test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_probs = model.predict_proba(X_test)[:, 1]

# Define business cost parameters
COST_FN = 5000   # Financial loss from an undetected loan default
COST_FP = 500    # Opportunity cost of rejecting a good applicant

thresholds = np.linspace(0.01, 0.99, 100)
costs = []

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    fn = np.sum((y_test == 1) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    costs.append(total_cost)

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
min_cost = costs[optimal_idx]

print("=" * 50)
print(f"Standard Threshold (0.50) Cost: ${costs[49]:,.2f}")
print(f"Optimal Business Cutoff       : {optimal_threshold:.2f}")
print(f"Optimized Financial Cost      : ${min_cost:,.2f}")
print(f"Expected Business Savings     : ${costs[49] - min_cost:,.2f}")
print("=" * 50)
