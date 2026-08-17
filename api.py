import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

class Applicant(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is live!"}

@app.post("/predict")
def predict_default(applicant: Applicant):
    data = pd.DataFrame([applicant.model_dump()])
    for col, le in label_encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])
    
    scaled = scaler.transform(data)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])

    return {
        "prediction": "Default" if pred == 1 else "No Default",
        "default_probability": round(prob, 4)
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    features_df = df.copy()
    if 'LoanID' in features_df.columns:
        features_df = features_df.drop(['LoanID'], axis=1)
    if 'Default' in features_df.columns:
        features_df = features_df.drop(['Default'], axis=1)

    for col, le in label_encoders.items():
        if col in features_df.columns:
            features_df[col] = le.transform(features_df[col])

    scaled = scaler.transform(features_df)
    df['Predicted_Default'] = model.predict(scaled)
    df['Default_Probability'] = model.predict_proba(scaled)[:, 1].round(4)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
