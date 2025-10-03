import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "customers.csv"
MODEL = Path(__file__).resolve().parents[2] / "model.pkl"

def main():
    df = pd.read_csv(DATA)
    X = df[["monthly_charges","tenure_months","is_promo","support_tickets_90d","late_payments_12m"]]
    y = df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    print(f"AUC: {auc:.3f}")
    joblib.dump(lr, MODEL)
    print(f"Saved {MODEL}")

if __name__ == "__main__":
    main()
