import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

df = pd.read_parquet("data/processed/features.parquet")

STEP = 30

# -------------------------
# STEP 1: Create future targets
# -------------------------
df["future_spread"] = df["spread"].shift(-STEP)
df["future_L10_log"] = df["L10_log"].shift(-STEP)
df["future_L50_log"] = df["L50_log"].shift(-STEP)
df["future_imbalance"] = df["Imbalance"].shift(-STEP)

target_cols = [
    "future_spread",
    "future_L10_log",
    "future_L50_log",
    "future_imbalance"
]

df = df.dropna(subset=target_cols).reset_index(drop=True)

# -------------------------
# STEP 2: TEMP SPLIT BEFORE FITTING QT
# -------------------------
train_end = int(len(df) * 0.70)
train_df = df.iloc[:train_end].copy()
full_df = df.copy()

# -------------------------
# STEP 3: FIT QT ONLY ON TRAIN
# -------------------------
qt = QuantileTransformer(
    n_quantiles=min(1000, len(train_df)),
    output_distribution="normal",
    random_state=42
)

qt.fit(train_df[target_cols])

# apply transform to full dataset
df[[
    "QT_future_spread",
    "QT_future_L10_log",
    "QT_future_L50_log",
    "QT_future_imbalance"
]] = qt.transform(df[target_cols])

# -------------------------
# STEP 4: Stress Score
# -------------------------
df["Future_Stress_Score"] = (
    df["QT_future_spread"]
    - df["QT_future_L10_log"]
    - df["QT_future_L50_log"]
    + df["QT_future_imbalance"].abs()
)

# -------------------------
# STEP 5: FINAL DATASET
# -------------------------
training_df = df[
    ["L10_log", "L50_log", "spread", "Imbalance", "Future_Stress_Score"]
].dropna()

training_df.to_parquet("data/processed/training_dataset.parquet", index=False)

print("Done. Final shape:", training_df.shape)