from locust import HttpUser, task, between
import random

class LoanPredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def test_predict_endpoint(self):
        payload = {
            "Age": random.randint(21, 65),
            "Income": random.randint(30000, 150000),
            "LoanAmount": random.randint(5000, 50000),
            "CreditScore": random.randint(550, 800),
            "MonthsEmployed": random.randint(12, 120),
            "NumCreditLines": random.randint(1, 10),
            "InterestRate": round(random.uniform(5.0, 20.0), 1),
            "LoanTerm": random.choice([12, 24, 36, 48, 60]),
            "DTIRatio": round(random.uniform(0.1, 0.6), 2),
            "Education": random.choice(["High School", "Bachelor's", "Master's", "PhD"]),
            "EmploymentType": random.choice(["Full-time", "Part-time", "Self-employed"]),
            "MaritalStatus": random.choice(["Single", "Married", "Divorced"]),
            "HasMortgage": random.choice(["Yes", "No"]),
            "HasDependents": random.choice(["Yes", "No"]),
            "LoanPurpose": random.choice(["Home", "Auto", "Education", "Business", "Other"]),
            "HasCoSigner": random.choice(["Yes", "No"])
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def test_metrics_endpoint(self):
        self.client.get("/metrics")
