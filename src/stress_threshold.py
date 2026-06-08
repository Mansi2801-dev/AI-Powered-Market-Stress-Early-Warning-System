import pandas as pd
import numpy as np

# Load training dataset (ONLY TRAIN DATA SHOULD BE USED FOR THRESHOLDS)
df = pd.read_parquet("data/processed/training_dataset.parquet")

# Remove NaNs safely
df = df.dropna(subset=["Future_Stress_Score"])

# Convert to numpy
train_scores = df["Future_Stress_Score"].values

# -----------------------------
# QUANTILE THRESHOLDS
# -----------------------------
q50 = np.quantile(train_scores, 0.50)
q90 = np.quantile(train_scores, 0.90)
q95 = np.quantile(train_scores, 0.95)
q99 = np.quantile(train_scores, 0.99)
q999 = np.quantile(train_scores, 0.999)

print("\n=== STRESS SCORE THRESHOLDS ===")
print("Median (50%):", q50)
print("90% :", q90)
print("95% :", q95)
print("99% :", q99)
print("99.9%:", q999)

# -----------------------------
# REGIME FUNCTION
# -----------------------------
def stress_regime(score):
    if score < q90:
        return "NORMAL"
    elif score < q95:
        return "WARNING"
    elif score < q99:
        return "HIGH STRESS"
    else:
        return "CRITICAL"

sample_scores = [q50, q90, q95, q99, q999]

print("\n=== SAMPLE REGIMES ===")
for s in sample_scores:
    print(f"Score: {s:.4f} -> {stress_regime(s)}")