"""
preprocessing.py — Clean, impute, scale, and engineer features for clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import CLEAN_DATA_FILE, MODELS_DIR, ID_COLUMN, FEATURE_COLUMNS


def preprocess(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline: clean → engineer → scale."""
    df = df.copy()

    # ── Drop ID column ─────────────────────────────────────────────────────────
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # ── Standardize column names ───────────────────────────────────────────────
    df.columns = df.columns.str.upper().str.strip().str.replace(" ", "_")

    # ── Impute missing values with median ──────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print(f"Imputed {missing_before} missing values with column medians")

    # ── Drop duplicates ────────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # ── Feature Engineering ────────────────────────────────────────────────────
    # Spend ratio: purchases relative to credit limit
    df["PURCHASES_TO_LIMIT_RATIO"] = (
        df["PURCHASES"] / (df["CREDIT_LIMIT"] + 1)
    ).round(4)

    # Cash advance ratio: proportion of balance that is cash advance
    df["CASH_ADVANCE_RATIO"] = (
        df["CASH_ADVANCE"] / (df["BALANCE"] + 1)
    ).round(4)

    # Payment behaviour: payments made vs minimum required
    df["PAYMENT_TO_MINIMUM_RATIO"] = (
        df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1)
    ).round(4)

    # Monthly spend rate
    df["MONTHLY_AVG_PURCHASE"] = (df["PURCHASES"] / df["TENURE"]).round(2)

    print(f"Engineered 4 new features")
    print(f"Final shape: {df.shape}")

    # ── Scale features ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    if save:
        df_scaled.to_csv(CLEAN_DATA_FILE, index=False)
        # Also save unscaled for interpretability
        df.to_csv(str(CLEAN_DATA_FILE).replace(".csv", "_unscaled.csv"), index=False)
        print(f"Saved scaled data  -> data/processed/cc_clean.csv")
        print(f"Saved unscaled data -> data/processed/cc_clean_unscaled.csv")

    return df_scaled, df   # return (scaled, unscaled)
