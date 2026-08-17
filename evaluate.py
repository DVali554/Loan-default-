import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Load pipeline artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

csv_file = 'Loan_default.csv' if os.path.exists('Loan_default.csv') else 'data/Loan_default.csv'
df = pd.read_csv(csv_file)
if 'LoanID' in df.columns:
    df = df.drop(['LoanID'], axis=1)

for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

X = scaler.transform(df.drop('Default', axis=1))
y = df['Default'].values
feature_names = df.drop('Default', axis=1).columns

# Holdout evaluation set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=300)
plt.close()

# 2. ROC & Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax1.set_title('Receiver Operating Characteristic (ROC)')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')

ax2.plot(recall, precision, color='#2ca02c', lw=2, label=f'PR AUC = {pr_auc:.3f}')
ax2.set_title('Precision-Recall Curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig('assets/roc_pr_curves.png', dpi=300)
plt.close()

# 3. Top Feature Importances Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 5))
plt.title('Top 10 Most Predictive Features')
plt.bar(range(10), importances[indices], align='center', color='#3366cc')
plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=300)
plt.close()

# 4. Save Metrics Summary to JSON
report = classification_report(y_test, y_pred, output_dict=True)
metrics_summary = {
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4),
    "accuracy": round(report['accuracy'], 4),
    "macro_f1": round(report['macro avg']['f1-score'], 4),
    "default_recall": round(report['1']['recall'], 4),
    "default_precision": round(report['1']['precision'], 4)
}

with open('assets/metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print("Evaluation complete. Charts and metrics saved to assets/")
print(json.dumps(metrics_summary, indent=2))
