import pandas as pd
import numpy as np

LOOKBACK = 60


df = pd.read_parquet("data/processed/labeled_data.parquet")


feature_cols = [
    "L10_log",
    "L50_log",
    "spread",
    "Imbalance"
]


target_col = "Future_Stress_Score"


df = df.dropna(subset=feature_cols + ["Future_Stress_Score"])

X_data = df[feature_cols].values
y_data = df[target_col].values


X, y = [], []

for i in range(LOOKBACK - 1, len(df)):
    X.append(X_data[i - LOOKBACK + 1 : i + 1])
    y.append(y_data[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)


np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("Sequence dataset created successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)