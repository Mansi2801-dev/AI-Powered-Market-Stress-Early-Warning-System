import os
import sys
from pathlib import Path
import pandas as pd

# 1. Resolve Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 2. Locate Raw CSV
raw_files = list(RAW_DIR.glob("*.csv"))
if not raw_files:
    print(f"[-] Error: No CSV files found in {RAW_DIR}")
    print("    Please download the order book CSV and place it inside data/raw/")
    sys.exit(1)

input_file = raw_files[0]
print(f"[+] Loading raw dataset from: {input_file.name}")
df = pd.read_csv(input_file)

# 3. Validation & Quality Checks
print(f"[+] Raw Shape: {df.shape}")
print(f"[+] Missing values:\n{df.isna().sum().sum()} total NaNs")
print(f"[+] Duplicate rows: {df.duplicated().sum()}")

if "time_exchange_minute" in df.columns:
    if not df["time_exchange_minute"].is_monotonic_increasing:
        print("[!] Timestamps not sorted chronologically. Sorting now...")
        df = df.sort_values("time_exchange_minute").reset_index(drop=True)
    print(f"[+] Time range: {df['time_exchange_minute'].iloc[0]} to {df['time_exchange_minute'].iloc[-1]}")

# 4. Drop non-feature metadata if present
clean_df = df.drop(columns=["symbol_id"], errors="ignore")

# 5. Save Clean Dataset
output_file = PROCESSED_DIR / "clean_dataset.parquet"
clean_df.to_parquet(output_file, index=False)
print(f"[+] Successfully saved clean dataset to: {output_file}")
print(f"[+] Final clean shape: {clean_df.shape}\n")