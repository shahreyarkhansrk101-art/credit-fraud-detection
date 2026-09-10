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

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/clean_data.py
python src/train_model.py
```

## Why not just accuracy?

With 99.8%+ of transactions being legitimate, a model that predicts
"legitimate" every time is 99.8% "accurate" and catches zero fraud.
This project reports precision, recall, F1 and AUC-PR (precision-recall
AUC), which are far more meaningful under severe class imbalance than
plain accuracy or even ROC-AUC.
