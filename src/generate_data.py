"""
generate_data.py
-----------------
Produces a synthetic transactions dataset that mirrors the schema of the
well-known Kaggle "Credit Card Fraud Detection" dataset
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):

    Time, V1, V2, ..., V28, Amount, Class

- Time      : seconds elapsed since the first transaction
- V1..V28   : anonymized PCA-transformed features
- Amount    : transaction amount
- Class     : 1 = fraud, 0 = legitimate

Fraud is simulated as a rare class (~0.17%), matching real-world card
fraud prevalence, with a distinct-but-overlapping feature distribution
so the classification problem is realistically hard (not trivially
separable).

If you have access to the real Kaggle dataset, just drop creditcard.csv
into data/raw/ and downstream scripts (clean_data.py, train_model.py)
will use it unchanged -- they only depend on the column schema, not on
how the file was produced.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "creditcard.csv"

N_SAMPLES = 50_000
FRAUD_RATE = 0.0017  # ~0.17%, matches real-world card fraud prevalence
N_PCA_FEATURES = 28
RANDOM_SEED = 42


def generate_dataset(n_samples: int = N_SAMPLES, fraud_rate: float = FRAUD_RATE,
                      seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(n_samples * fraud_rate)))
    n_legit = n_samples - n_fraud

    # Legit transactions: PCA features ~ standard normal (as in the real dataset)
    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, N_PCA_FEATURES))
    # Fraud transactions: shifted mean + higher variance on a subset of
    # components, simulating the kind of anomalies PCA would surface.
    fraud_v = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, N_PCA_FEATURES))
    shifted_components = rng.choice(N_PCA_FEATURES, size=8, replace=False)
    fraud_v[:, shifted_components] += rng.normal(loc=3.0, scale=1.5,
                                                   size=(n_fraud, len(shifted_components)))

    # Amount: legit purchases mostly small/medium; fraud skews toward
    # either very small (card testing) or unusually large amounts.
    legit_amount = np.round(rng.gamma(shape=2.0, scale=40.0, size=n_legit), 2)
    fraud_amount = np.round(
        np.concatenate([
            rng.uniform(0, 5, size=n_fraud // 2),
            rng.uniform(200, 2000, size=n_fraud - n_fraud // 2),
        ]), 2
    )
    rng.shuffle(fraud_amount)

    # Time: seconds over a 2-day window, fraud slightly more common at
    # low-traffic (night) hours.
    two_days_seconds = 2 * 24 * 3600
    legit_time = rng.uniform(0, two_days_seconds, size=n_legit)
    night_biased = rng.normal(loc=two_days_seconds * 0.15, scale=two_days_seconds * 0.1,
                               size=n_fraud)
    fraud_time = np.clip(night_biased, 0, two_days_seconds)

    v_cols = [f"V{i}" for i in range(1, N_PCA_FEATURES + 1)]

    legit_df = pd.DataFrame(legit_v, columns=v_cols)
    legit_df["Time"] = legit_time
    legit_df["Amount"] = legit_amount
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=v_cols)
    fraud_df["Time"] = fraud_time
    fraud_df["Amount"] = fraud_amount
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)

    # Reorder columns to match the real dataset: Time, V1..V28, Amount, Class
    df = df[["Time"] + v_cols + ["Amount", "Class"]]

    # Inject a small amount of realistic messiness for the cleaning step
    # downstream to actually have something to do:
    dup_idx = rng.choice(df.index, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)  # duplicate rows

    missing_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[missing_idx, "Amount"] = np.nan  # missing values

    neg_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[neg_idx, "Amount"] = -df.loc[neg_idx, "Amount"]  # bad negative amounts

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_FILE}")
    print(f"Fraud cases: {int(df['Class'].sum())} ({df['Class'].mean() * 100:.3f}%)")


if __name__ == "__main__":
    main()
