import pandas as pd
import numpy as np

LOOKBACK = 60

df = pd.read_parquet("data/processed/training_dataset.parquet")

feature_cols = [
    "L10_log",
    "L50_log",
    "spread",
    "Imbalance"
]

X_data = df[feature_cols].values
y_data = df["Future_Stress_Score"].values

X = []
y = []

for i in range(LOOKBACK - 1, len(df)):
    X.append(X_data[i - LOOKBACK + 1 : i + 1])
    y.append(y_data[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("\nSequence dataset saved successfully.")