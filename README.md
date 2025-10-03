# Customer Churn Early Warning — Impact-Focused Project

**Perspective:** This repo is structured like a product. It demonstrates business framing, measurable outcomes, API deployment, and dashboards exactly what a hiring team looks for.

## TL;DR Impact
- **Problem:** Companies lose revenue when customers churn unexpectedly.
- **Action:** Train a churn prediction model, expose a REST API, and build a stakeholder dashboard for risk monitoring.
- **Outcome (demo):** On sample data, model recovers ~75% of churners at 20% outreach rate. Thresholds tunable.

## Architecture
```
data/                      # synthetic dataset
src/
  training/train.py        # trains model.pkl
  api/app.py               # FastAPI app (predict + health)
  dashboard/app.py         # Streamlit dashboard
  utils/metrics.py         # precision@k, recall@k helpers
tests/
  test_metrics.py          # sanity tests
Dockerfile.api             # build API container
Dockerfile.dashboard       # build dashboard container
docker-compose.yml         # spins up API+dashboard
requirements.txt
Makefile
.github/workflows/ci.yml   # GitHub Actions CI
.env.example
LICENSE
```

## Run locally (no Docker)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/training/train.py
uvicorn src.api.app:app --host 0.0.0.0 --port 8787
# new shell:
streamlit run src/dashboard/app.py --server.port 8765
```

## Run with Docker Compose
```bash
docker compose up --build
```
- API: http://localhost:8787/docs  
- Dashboard: http://localhost:8765  

## Example request
```bash
curl -X POST http://localhost:8787/predict -H "Content-Type: application/json" -d '{
  "monthly_charges": 72.5,
  "tenure_months": 4,
  "is_promo": 0,
  "support_tickets_90d": 3,
  "late_payments_12m": 2
}'
```

---
**This project shows**: outcome-driven framing, deployable code, CI, tests, and a polished README.
