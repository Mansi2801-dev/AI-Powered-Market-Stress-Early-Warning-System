from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

input_file = PROCESSED_DIR / "clean_dataset.parquet"
if not input_file.exists():
    raise FileNotFoundError(f"Missing {input_file}. Please run preprocess.py first.")

print(f"[+] Loading: {input_file}")
df = pd.read_parquet(input_file)

# 1. Spread Calculation (Top-of-Book)
df["spread"] = df["asks[0].price"] - df["bids[0].price"]

# 2. Cumulative Liquidity (Top 10 and Top 50 levels)
ask_cols_10 = [f"asks[{i}].size" for i in range(10)]
bid_cols_10 = [f"bids[{i}].size" for i in range(10)]

ask_cols_50 = [f"asks[{i}].size" for i in range(50)]
bid_cols_50 = [f"bids[{i}].size" for i in range(50)]

df["L10_bids"] = df[bid_cols_10].sum(axis=1)
df["L10_asks"] = df[ask_cols_10].sum(axis=1)
df["L10_total"] = df["L10_bids"] + df["L10_asks"]

df["L50_bids"] = df[bid_cols_50].sum(axis=1)
df["L50_asks"] = df[ask_cols_50].sum(axis=1)
df["L50_total"] = df["L50_bids"] + df["L50_asks"]

# Log transform for liquidity stabilization
df["L10_log"] = np.log1p(df["L10_total"])
df["L50_log"] = np.log1p(df["L50_total"])

# 3. Order Book Imbalance (with zero-division guard)
denom = df["L10_bids"] + df["L10_asks"]
df["Imbalance"] = np.where(denom > 0, (df["L10_bids"] - df["L10_asks"]) / denom, 0.0)

# 4. Retain only necessary columns
feature_cols = ["spread", "L10_log", "L50_log", "Imbalance"]
if "time_exchange_minute" in df.columns:
    output_df = df[["time_exchange_minute"] + feature_cols].copy()
else:
    output_df = df[feature_cols].copy()

output_file = PROCESSED_DIR / "features.parquet"
output_df.to_parquet(output_file, index=False)

print("\n--- Features Summary ---")
for col in feature_cols:
    print(f"{col:10s} -> Min: {output_df[col].min():.5f} | Mean: {output_df[col].mean():.5f} | Max: {output_df[col].max():.5f}")

print(f"\n[+] Features dataset saved successfully: {output_file}")
print(f"[+] Output shape: {output_df.shape}\n")