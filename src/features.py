import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("data/processed/clean_dataset.parquet")

df["spread"] = df["asks[0].price"] - df["bids[0].price"]

print("Minimum spread:", df["spread"].min())
print("Maximum spread:", df["spread"].max())

ask_cols = [f"asks[{i}].size" for i in range(10)]
bid_cols = [f"bids[{i}].size" for i in range(10)]

ask_cols_50 = [f"asks[{i}].size" for i in range(50)]
bid_cols_50 = [f"bids[{i}].size" for i in range(50)]

df["L10_bids"] = df[bid_cols].sum(axis=1)
df["L10_asks"] = df[ask_cols].sum(axis=1)
df["L10_total"] = df["L10_bids"] + df["L10_asks"]

df["L50_bids"] = df[bid_cols_50].sum(axis=1)
df["L50_asks"] = df[ask_cols_50].sum(axis=1)
df["L50_total"] = df["L50_bids"] + df["L50_asks"]

df["L10_log"] = np.log1p(df["L10_total"])
df["L50_log"] = np.log1p(df["L50_total"])
print("\nL50 Liquidity Statistics")
print("Min L50_log :", df["L50_log"].min())
print("Max L50_log :", df["L50_log"].max())
print("Mean L50_log:", df["L50_log"].mean())


df["Imbalance"] = (df["L10_bids"] - df["L10_asks"])/(df["L10_bids"] + df["L10_asks"])
print("Min Imbalance: ", df["Imbalance"].min())
print("Max Imbalance: ", df["Imbalance"].max())
print("Mean Imbalance: ", df["Imbalance"].mean())

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_dir / "features.parquet", index=False)
print("\nFeatures dataset saved successfully.")
print("Shape:", df.shape)
