"""
FastAPI inference service for the credit card fraud detection model.

Endpoints:
    GET  /health          -- liveness check
    GET  /model-info       -- metadata about the loaded model
    GET  /sample?label=fraud|legit  -- a real row pulled from the held-out
                                        test set, for demo purposes
    POST /predict          -- run the model on a given transaction
"""

from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test.csv"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves predictions from the trained fraud detection model.",
    version="1.0.0",
)

# Allow the Next.js dev server (and any origin, for demo purposes) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_test_df = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run the training pipeline first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_test_df():
    global _test_df
    if _test_df is None:
        if not TEST_DATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Test data not found at {TEST_DATA_PATH}. Run the pipeline first.",
            )
        _test_df = pd.read_csv(TEST_DATA_PATH)
    return _test_df


class Transaction(BaseModel):
    """A single transaction to score. V1..V28 are the anonymized PCA
    features from the dataset; Time and Amount are pre-scaled the same
    way the training data was (use /sample to get a ready-to-use example)."""

    Time: float
    Amount: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_name: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    model = get_model()
    return {
        "model_type": type(model).__name__,
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }


@app.get("/sample")
def sample(label: Optional[Literal["fraud", "legit"]] = Query(default=None)):
    """Return a real transaction from the held-out test set, so the demo
    UI has something realistic to send to /predict without hand-typing
    28 PCA feature values."""
    df = get_test_df()
    if label == "fraud":
        pool = df[df["Class"] == 1]
    elif label == "legit":
        pool = df[df["Class"] == 0]
    else:
        pool = df

    if pool.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for label={label}")

    row = pool.sample(1).iloc[0]
    transaction = {col: float(row[col]) for col in FEATURE_COLUMNS}
    return {
        "transaction": transaction,
        "true_label": "fraud" if int(row["Class"]) == 1 else "legit",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = get_model()
    X = pd.DataFrame([transaction.model_dump()])[FEATURE_COLUMNS]
    proba = float(model.predict_proba(X)[0, 1])
    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= 0.5,
        model_name=type(model).__name__,
    )
