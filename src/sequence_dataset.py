from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

input_file = PROCESSED_DIR / "training_dataset.parquet"
if not input_file.exists():
    raise FileNotFoundError(f"Missing {input_file}. Please run labels.py first.")

df = pd.read_parquet(input_file)

feature_cols = ["L10_log", "L50_log", "spread", "Imbalance"]
target_col = "Future_Stress_Score"

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

# 1. Feature Normalization (Fit strictly on first 70% to avoid leakage)
train_end = int(len(df) * 0.70)
scaler = StandardScaler()
scaler.fit(df.iloc[:train_end][feature_cols])

# Save scaler for inference dashboard
joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
print(f"[+] Fitted StandardScaler and saved to: {MODELS_DIR / 'feature_scaler.joblib'}")

# Transform all rows
X_raw = scaler.transform(df[feature_cols]).astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)

# 2. Build 60-step Rolling Sequences
LOOKBACK = 60
num_samples = len(df) - LOOKBACK + 1

print(f"[+] Generating {num_samples} sliding sequences of shape ({LOOKBACK}, {len(feature_cols)})...")

# Fast vector window construction
X = np.lib.stride_tricks.sliding_window_view(X_raw, (LOOKBACK, len(feature_cols))).squeeze(1)
# Targets correspond to the state predicted from the end of the lookback window
y = y_raw[LOOKBACK - 1 :]

X = np.ascontiguousarray(X, dtype=np.float32)
y = np.ascontiguousarray(y, dtype=np.float32)

# 3. Save sequence arrays (saving both y.npy and Y.npy for compatibility)
np.save(PROCESSED_DIR / "X.npy", X)
np.save(PROCESSED_DIR / "Y.npy", y)
np.save(PROCESSED_DIR / "y.npy", y)

print(f"[+] Sequence dataset created successfully!")
print(f"[+] X shape: {X.shape} (dtype: {X.dtype})")
print(f"[+] Y shape: {y.shape} (dtype: {y.dtype})\n")