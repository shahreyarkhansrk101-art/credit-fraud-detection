# Credit Card Fraud Detection

A binary classification project that detects fraudulent credit card
transactions in a highly imbalanced dataset (~0.17% fraud rate),
modeled on the structure of the well-known Kaggle
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Problem

Card issuers need to flag fraudulent transactions in near real time
without generating so many false alarms that legitimate customers get
blocked. This is a classic extreme class-imbalance problem: naive
accuracy is meaningless (predicting "not fraud" for everything scores
>99.8%), so the project is built around precision/recall and AUC-PR
instead.

## Project structure

```
credit-fraud-detection/
├── data/
│   ├── raw/            # original/generated data
│   └── processed/      # cleaned, model-ready data
├── src/
│   ├── generate_data.py  # data acquisition
│   ├── clean_data.py     # cleaning + preprocessing
│   └── train_model.py    # model training + evaluation
├── models/              # saved trained model(s)
├── reports/figures/     # evaluation plots
├── api/                 # FastAPI inference service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # Next.js demo UI
│   ├── app/page.tsx
│   └── Dockerfile
├── Dockerfile            # training pipeline image
├── docker-compose.yml    # orchestrates pipeline + api + frontend
└── requirements.txt
```

## Pipeline

1. **Data preparation** (`src/generate_data.py`) — produces
   `data/raw/creditcard.csv` with the same schema as the real Kaggle
   dataset (`Time, V1..V28, Amount, Class`). If you have access to the
   real dataset, just drop it in `data/raw/creditcard.csv` instead —
   everything downstream works unchanged.
2. **Data cleaning** (`src/clean_data.py`) — deduplication, missing
   value handling, invalid-value correction, scaling, train/test split.
3. **Modeling** (`src/train_model.py`) — trains and compares baseline
   Logistic Regression vs. Random Forest with class-weighting to
   handle the imbalance, evaluates with precision, recall, F1,
   ROC-AUC and AUC-PR, and saves the best model.

## Usage

### Option A: Local (Python)

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/clean_data.py
python src/train_model.py
```

### Option B: Docker

Build and run the full pipeline (generate → clean → train) in one shot,
with results written back to your local `data/`, `models/`, and
`reports/` folders:

```bash
docker compose up --build
```

Or with plain `docker` (no compose):

```bash
docker build -t fraud-detection .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  fraud-detection
```

To run just one stage instead of the whole pipeline (e.g. after
swapping in the real Kaggle dataset and only wanting to re-clean +
retrain):

```bash
docker compose run --rm pipeline python src/clean_data.py
docker compose run --rm pipeline python src/train_model.py
```

## Full stack: API + demo UI

There's a FastAPI service that serves predictions from the trained
model, and a small Next.js page to demo it live — pull a real
transaction from the test set, run it through the model, and see
whether the prediction matches the true label.

**With Docker (recommended):**

```bash
docker compose up --build
```

This builds and runs all three services in order: the pipeline trains
the model first, then the API and frontend start. Once it settles,
open:

- **http://localhost:3000** — the demo UI
- **http://localhost:8000/docs** — interactive API docs (Swagger UI)

**Without Docker (running locally):**

```bash
# 1. Make sure you've already run the pipeline (see above) so
#    models/best_model.joblib and data/processed/test.csv exist.

# 2. Start the API
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000

# 3. In another terminal, start the frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000.

### API endpoints

| Method | Path          | Description                                  |
|--------|---------------|-----------------------------------------------|
| GET    | `/health`     | Liveness check                                |
| GET    | `/model-info` | Model type and expected feature names         |
| GET    | `/sample?label=fraud\|legit` | A real transaction from the test set |
| POST   | `/predict`    | Score a transaction, returns fraud probability |


## Why not just accuracy?

With 99.8%+ of transactions being legitimate, a model that predicts
"legitimate" every time is 99.8% "accurate" and catches zero fraud.
This project reports precision, recall, F1 and AUC-PR (precision-recall
AUC), which are far more meaningful under severe class imbalance than
plain accuracy or even ROC-AUC.

## Results

| Model               | ROC-AUC | AUC-PR |
|---------------------|---------|--------|
| Logistic Regression | 0.924   | 0.164  |
| Random Forest        | 0.999   | 0.744  |

Random Forest ranks transactions far better (AUC-PR 0.74 vs 0.16), but
at the default 0.5 probability threshold it predicts "not fraud" for
every case, so its raw recall looks worse than Logistic Regression's.
Logistic Regression, on the other hand, catches most fraud (76% recall)
but at the cost of huge numbers of false positives.

This is a genuine, common failure mode in fraud detection: a
well-ranking model can still perform badly at a naive 0.5 cutoff.
The natural next step (good discussion point for a presentation) is
threshold tuning against the precision-recall curve in
`reports/figures/precision_recall_curve.png`, rather than relying on
the default classification threshold.
