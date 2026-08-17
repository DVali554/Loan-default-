# 🏦 Loan Default Prediction System

An end-to-end Machine Learning pipeline that predicts borrower default risk using financial indicators, demographic attributes, and loan metadata.

## 📌 Features
- **Data Preprocessing**: Categorical encoding with LabelEncoder and feature scaling with StandardScaler.
- **Imbalance Handling**: Applies SMOTE oversampling to balance default vs. non-default classes.
- **Model Architecture**: Evaluates credit risk using a tuned RandomForestClassifier.
- **Interactive UI**: Real-time prediction dashboard built with Streamlit.

## 🛠️ Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Run Predictions
```bash
# Terminal inference
python predict.py

# Launch interactive Web App
streamlit run app.py
```
