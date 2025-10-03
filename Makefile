SHELL := /bin/bash

.PHONY: setup train api dash test clean

setup:
	python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

train:
	python src/training/train.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8787

dash:
	streamlit run src/dashboard/app.py --server.port 8765

test:
	pytest -q

clean:
	rm -f model.pkl
