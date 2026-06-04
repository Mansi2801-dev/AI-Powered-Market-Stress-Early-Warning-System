import pandas as pd

df = pd.read_parquet("data/processed/clean_dataset.parquet")
df["spread"] = df["asks[0].price"] - df["bids[0].price"]
print("minimum spread:", df["spread"].min())
print("maximum spread:", df["spread"].max())