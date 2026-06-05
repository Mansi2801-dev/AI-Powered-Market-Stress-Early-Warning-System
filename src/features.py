import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/clean_dataset.parquet")

df["spread"] = df["asks[0].price"] - df["bids[0].price"]

print("Minimum spread:", df["spread"].min())
print("Maximum spread:", df["spread"].max())

ask_cols = [f"asks[{i}].size" for i in range(10)]
bid_cols = [f"bids[{i}].size" for i in range(10)]

df["L10_bids"] = df[bid_cols].sum(axis=1)
df["L10_asks"] = df[ask_cols].sum(axis=1)
df["L10_total"] = df["L10_bids"] + df["L10_asks"]

df["L10_log"] = np.log1p(df["L10_total"])
print("\nL10 Liquidity Statistics")
print("Min L10_log :", df["L10_log"].min())
print("Max L10_log :", df["L10_log"].max())
print("Mean L10_log:", df["L10_log"].mean())


df["Imbalance"] = (df["L10_bids"] - df["L10_asks"])/(df["L10_bids"] + df["L10_asks"])
print("Min Imbalance: ", df["Imbalance"].min())
print("Max Imbalance: ", df["Imbalance"].max())
print("Mean Imbalance: ", df["Imbalance"].mean())
