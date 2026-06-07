import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# Load dataset
df = pd.read_parquet("data/processed/features.parquet")

STEP = 30

df["future_spread"] = df["spread"].shift(-STEP)
df["future_L10_log"] = df["L10_log"].shift(-STEP)
df["future_L50_log"] = df["L50_log"].shift(-STEP)
df["future_imbalance"] = df["Imbalance"].shift(-STEP)

print("--- Raw Future Statistics ---")
for col in ["future_spread", "future_L10_log", "future_L50_log", "future_imbalance"]:
    print(f"\n{col}")
    print("Mean:", df[col].mean())
    print("Std :", df[col].std())

n_samples = df.dropna(subset=["future_spread", "future_L10_log", "future_L50_log", "future_imbalance"]).shape[0]
n_quantiles = min(1000, n_samples)

qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=42)

target_cols = ["future_spread", "future_L10_log", "future_L50_log", "future_imbalance"]
transformed_cols = ["QT_future_spread", "QT_future_L10_log", "QT_future_L50_log", "QT_future_imbalance"]

valid_idx = df[target_cols].dropna().index
df.loc[valid_idx, transformed_cols] = qt.fit_transform(df.loc[valid_idx, target_cols])

print("\n--- Quantile Transformed Statistics ---")
for col in transformed_cols:
    print(f"\n{col}")
    print("Mean:", df[col].mean())
    print("Std :", df[col].std())

df["Future_Stress_Score"] = (
      df["QT_future_spread"]
    - df["QT_future_L10_log"]
    - df["QT_future_L50_log"]
    + df["QT_future_imbalance"].abs()
)

print("\n--- Future Stress Score ---")
print("Min :", df["Future_Stress_Score"].min())
print("Max :", df["Future_Stress_Score"].max())
print("Mean:", df["Future_Stress_Score"].mean())
print("Std :", df["Future_Stress_Score"].std())

print("\n--- Quantiles of Stress Score ---")
print("90% :", df["Future_Stress_Score"].quantile(0.90))
print("95% :", df["Future_Stress_Score"].quantile(0.95))
print("99% :", df["Future_Stress_Score"].quantile(0.99))
print("99.9%:", df["Future_Stress_Score"].quantile(0.999))

idx = df["Future_Stress_Score"].idxmax()

print("\n--- Maximum Stress Event Details ---")
print(df.loc[idx, [
    "future_spread",
    "QT_future_spread",
    "QT_future_L10_log",
    "QT_future_L50_log",
    "QT_future_imbalance",
    "Future_Stress_Score"
]])

training_df = df[
    [
        "L10_log",
        "L50_log",
        "spread",
        "Imbalance",
        "Future_Stress_Score"
    ]
].copy()

training_df = training_df.dropna(subset=["Future_Stress_Score"])

training_df.to_parquet(
    "data/processed/training_dataset.parquet",
    index=False
)

print("\n--- Training Dataset ---")
print("Shape:", training_df.shape)
print(training_df.head())
