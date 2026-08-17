import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# 1. Load baseline training dataset
csv_path = 'Loan_default.csv' if pd.io.common.file_exists('Loan_default.csv') else 'data/Loan_default.csv'
df_train = pd.read_csv(csv_path)

# 2. Load recent production inferences from SQLite
conn = sqlite3.connect('predictions.db')
df_prod = pd.read_sql_query("SELECT * FROM prediction_logs", conn)
conn.close()

numeric_features = ['age', 'income', 'loan_amount', 'credit_score', 'dti_ratio']

print("=" * 60)
print("🔍 DATA DRIFT AUDIT REPORT (Kolmogorov-Smirnov Test)")
print("=" * 60)

if len(df_prod) < 10:
    print(f"⚠️ Insufficient production logs ({len(df_prod)} records).")
    print("Generating synthetic incoming traffic for drift demonstration...\n")
    # Simulate a shifted production batch (e.g., lower credit scores & higher DTI)
    np.random.seed(42)
    df_prod = pd.DataFrame({
        'age': np.random.normal(30, 8, 200),
        'income': np.random.normal(55000, 15000, 200),
        'loan_amount': np.random.normal(28000, 8000, 200),
        'credit_score': np.random.normal(610, 50, 200),
        'dti_ratio': np.random.normal(0.45, 0.1, 200)
    })

# Run 2-sample KS test per numeric column
drift_detected = False
ALPHA = 0.05

for feat in numeric_features:
    # Match column casing between training and production
    train_col = [c for c in df_train.columns if c.lower() == feat.lower()][0]
    stat, p_value = ks_2samp(df_train[train_col], df_prod[feat])
    
    status = "🚨 DRIFT DETECTED" if p_value < ALPHA else "✅ STABLE"
    if p_value < ALPHA:
        drift_detected = True
        
    print(f"{feat.upper():<16} | KS Stat: {stat:.4f} | p-value: {p_value:.4e} | {status}")

print("=" * 60)
if drift_detected:
    print("⚠️ Recommendation: Retrain model on recent production window.")
else:
    print("✨ Status: Feature distributions remain stable. No retraining required.")
print("=" * 60)
