import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, confloat, conint

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))

class CustomerFeatures(BaseModel):
    monthly_charges: confloat(ge=0)
    tenure_months: conint(ge=0)
    is_promo: conint(ge=0, le=1)
    support_tickets_90d: conint(ge=0)
    late_payments_12m: conint(ge=0)

app = FastAPI(title="Churn Early Warning API", version="1.0.0")

_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok", "threshold": RISK_THRESHOLD}

@app.post("/predict")
def predict(f: CustomerFeatures):
    model = load_model()
    x = np.array([[
        f.monthly_charges, f.tenure_months, f.is_promo,
        f.support_tickets_90d, f.late_payments_12m
    ]])
    proba = float(model.predict_proba(x)[0,1])
    is_risk = proba >= RISK_THRESHOLD
    return {"risk_score": proba, "is_at_risk": bool(is_risk), "threshold": RISK_THRESHOLD}
