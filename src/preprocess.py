import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/ADA_USDT.csv")
first = df['time_exchange_minute'].iloc[0]
last = df['time_exchange_minute'].iloc[-1]
sort = df['time_exchange_minute'].is_monotonic_increasing

print(df.shape)
print(df.isna().sum())
print(df.duplicated().sum())
print(f"first timestamp is {first} and last is {last}")
print(sort)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
processed_df = df.drop(columns=["symbol_id"])
processed_df.to_parquet(output_dir / "clean_dataset.parquet", index=False)

