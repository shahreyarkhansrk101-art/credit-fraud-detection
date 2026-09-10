"""
train_model.py
---------------
Trains and compares two classifiers on the imbalanced fraud dataset:

    1. Logistic Regression (class_weight="balanced") -- fast, interpretable baseline
    2. Random Forest (class_weight="balanced")        -- stronger, non-linear model

Because fraud is ~0.17% of the data, plain accuracy is a misleading
metric (predicting "no fraud" for everything scores ~99.8%). We
evaluate with precision, recall, F1, ROC-AUC and average precision
(AUC-PR), and select the model with the higher AUC-PR.

Outputs:
    models/best_model.joblib
    reports/figures/confusion_matrix.png
    reports/figures/precision_recall_curve.png
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RANDOM_SEED = 42


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]
    return X_train, y_train, X_test, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=3))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUC-PR (Average Precision): {ap:.4f}")

    return {"model": model, "name": name, "roc_auc": roc_auc, "auc_pr": ap,
             "y_pred": y_pred, "y_proba": y_proba}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=RANDOM_SEED)
    log_reg.fit(X_train, y_train)
    lr_result = evaluate("Logistic Regression", log_reg, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_result = evaluate("Random Forest", rf, X_test, y_test)

    # Pick the better model by AUC-PR, the most informative metric here
    best = max([lr_result, rf_result], key=lambda r: r["auc_pr"])
    print(f"\nBest model: {best['name']} (AUC-PR={best['auc_pr']:.4f})")

    joblib.dump(best["model"], MODELS_DIR / "best_model.joblib")
    print(f"Saved best model to {MODELS_DIR / 'best_model.joblib'}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred"], display_labels=["Legit", "Fraud"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {best['name']}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve for both models
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, lr_result["y_proba"],
                                             name="Logistic Regression", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, rf_result["y_proba"],
                                             name="Random Forest", ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
