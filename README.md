# 🏦 Loan Default Prediction & Explainability Platform

[![CI Pipeline](https://github.com/DVali554/Loan-default-/actions/workflows/ci.yml/badge.svg)](https://github.com/DVali554/Loan-default-/actions)

An end-to-end production Machine Learning pipeline for credit risk assessment, featuring model explainability (SHAP), automated CI/CD testing, batch inference, and interactive web dashboards.

---

## 📈 Model Performance & Visual Evaluation

| Confusion Matrix | ROC & PR Curves |
| :---: | :---: |

### 🔍 Top Feature Importances

---

## 🏗️ System Architecture

```text
[ Raw Loan Data ] 
       │
       ▼
[ Preprocessing & SMOTE Oversampling ]
       │
       ▼
[ Tuned Random Forest Classifier ]
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
 [ FastAPI Backend ]    [ Streamlit Web App ]    [ SHAP Explainability ]
  - Single JSON           - Sliders & Inputs       - Feature Attributions
  - Batch CSV Streaming   - Real-time Inference    - Risk Factor Plots
  - SQLite Audit Logging  - Live Audit Analytics
```

---

## 🛠️ Quickstart

### 1. Installation
```bash
git clone https://github.com/DVali554/Loan-default-.git
cd Loan-default-
pip install -r requirements.txt
```

### 2. Train and Evaluate
```bash
# Train model artifacts
python train.py

# Generate evaluation charts and metrics
python evaluate.py
```

### 3. Launch Applications
```bash
# Interactive UI + SHAP Visualizations
streamlit run app.py

# REST API with Swagger Docs
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing & CI
Continuous Integration runs automatically via GitHub Actions on every pull request and push to `main`:
```bash
pytest -v
```