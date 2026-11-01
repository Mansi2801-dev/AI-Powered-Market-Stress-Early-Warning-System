from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

input_file = PROCESSED_DIR / "features.parquet"
if not input_file.exists():
    raise FileNotFoundError(f"Missing {input_file}. Please run features.py first.")

df = pd.read_parquet(input_file)

# 1. Project 30 steps into the future
STEP = 30
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

# 2. Strict Chronological Split for Scaling (First 70% only)
train_end = int(len(df) * 0.70)
train_df = df.iloc[:train_end]

qt = QuantileTransformer(
    n_quantiles=min(1000, len(train_df)),
    output_distribution="normal",
    random_state=42
)
qt.fit(train_df[target_cols])

# Save target transformer for future interpretability
joblib.dump(qt, MODELS_DIR / "target_scaler.joblib")

# 3. Transform targets for full dataset
transformed = qt.transform(df[target_cols])
df["QT_future_spread"] = transformed[:, 0]
df["QT_future_L10_log"] = transformed[:, 1]
df["QT_future_L50_log"] = transformed[:, 2]
df["QT_future_imbalance"] = transformed[:, 3]

# 4. Composite Stress Score:
# High Spread (+) + Low Liquidity (-) + Extreme Imbalance (+)
df["Future_Stress_Score"] = (
    df["QT_future_spread"]
    - df["QT_future_L10_log"]
    - df["QT_future_L50_log"]
    + df["QT_future_imbalance"].abs()
)

# 5. Export training dataset
export_cols = ["L10_log", "L50_log", "spread", "Imbalance", "Future_Stress_Score"]
training_df = df[export_cols].dropna().reset_index(drop=True)

output_file = PROCESSED_DIR / "training_dataset.parquet"
training_df.to_parquet(output_file, index=False)

print(f"[+] Generated Future Stress Labels (+{STEP} steps)")
print(f"[+] Future_Stress_Score summary: Mean={training_df['Future_Stress_Score'].mean():.4f}, Std={training_df['Future_Stress_Score'].std():.4f}")
print(f"[+] Saved to: {output_file} | Shape: {training_df.shape}\n")