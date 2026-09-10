"""
clean_data.py
-------------
Cleans the raw transactions data and prepares it for modeling:

1. Drop exact duplicate rows.
2. Handle missing values (impute Amount with median).
3. Fix invalid values (negative Amount -> take absolute value; a
   transaction amount can't be negative).
4. Scale Time and Amount (the only two "human-readable" features --
   V1..V28 are already PCA components and roughly standardized).
5. Stratified train/test split (important: with ~0.17% fraud, a random
   split can easily starve the test set of fraud examples unless it's
   stratified on the target).

Outputs:
    data/processed/train.csv
    data/processed/test.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEST_SIZE = 0.2
RANDOM_SEED = 42


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # 1. Drop exact duplicates
    df = df.drop_duplicates()
    n_after_dedup = len(df)

    # 2. Missing values: impute Amount with the median (robust to the
    #    heavy right-skew typical of transaction amounts)
    n_missing = df["Amount"].isna().sum()
    df["Amount"] = df["Amount"].fillna(df["Amount"].median())

    # 3. Invalid values: a transaction amount can't be negative
    n_negative = (df["Amount"] < 0).sum()
    df["Amount"] = df["Amount"].abs()

    # Sanity checks
    assert df["Class"].isin([0, 1]).all(), "Class column must be binary"
    assert df.isna().sum().sum() == 0, "Unexpected missing values remain"

    print(f"Rows before cleaning:      {n_before:,}")
    print(f"Duplicates removed:        {n_before - n_after_dedup:,}")
    print(f"Missing Amount imputed:    {n_missing:,}")
    print(f"Negative Amount corrected: {n_negative:,}")
    print(f"Rows after cleaning:       {len(df):,}")

    return df.reset_index(drop=True)


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Fit the scaler on train only, to avoid leaking test-set info
    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    train_df = X_train.copy()
    train_df["Class"] = y_train.values
    test_df = X_test.copy()
    test_df["Class"] = y_test.values

    print(f"\nTrain set: {len(train_df):,} rows ({train_df['Class'].mean() * 100:.3f}% fraud)")
    print(f"Test set:  {len(test_df):,} rows ({test_df['Class'].mean() * 100:.3f}% fraud)")

    return train_df, test_df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df_clean = clean(df)
    train_df, test_df = split_and_scale(df_clean)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nSaved processed data to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
