import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

# Default fallback thresholds (based on target distribution)
DEFAULT_THRESHOLDS = {
    "q50": 0.50,
    "q90": 2.20,
    "q95": 3.10,
    "q99": 4.50
}

def load_thresholds():
    """Loads saved quantile thresholds, or returns defaults."""
    if THRESHOLDS_PATH.exists():
        try:
            with open(THRESHOLDS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_THRESHOLDS

def stress_regime(score: float, thresholds: dict = None) -> str:
    """
    Categorizes a continuous stress score into a market regime:
    - NORMAL      : Below 90th percentile (typical market condition)
    - WARNING     : 90th to 95th percentile (emerging liquidity stress)
    - HIGH STRESS : 95th to 99th percentile (severe depth imbalance/spread widening)
    - CRITICAL    : Top 1% (> 99th percentile, flash crash risk)
    """
    th = thresholds or load_thresholds()
    if score < th["q90"]:
        return "NORMAL"
    elif score < th["q95"]:
        return "WARNING"
    elif score < th["q99"]:
        return "HIGH STRESS"
    else:
        return "CRITICAL"

if __name__ == "__main__":
    train_y_path = PROCESSED_DIR / "Y_train.npy"
    dataset_path = PROCESSED_DIR / "training_dataset.parquet"

    if train_y_path.exists():
        scores = np.load(train_y_path)
    elif dataset_path.exists():
        df = pd.read_parquet(dataset_path)
        train_end = int(len(df) * 0.70)
        scores = df.iloc[:train_end]["Future_Stress_Score"].values
    else:
        raise FileNotFoundError("Neither Y_train.npy nor training_dataset.parquet found. Run labels.py first.")

    q50 = float(np.quantile(scores, 0.50))
    q90 = float(np.quantile(scores, 0.90))
    q95 = float(np.quantile(scores, 0.95))
    q99 = float(np.quantile(scores, 0.99))

    threshold_dict = {"q50": q50, "q90": q90, "q95": q95, "q99": q99}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(threshold_dict, f, indent=4)

    print("\n=== DERIVED STRESS SCORE THRESHOLDS (TRAIN SET) ===")
    print(f"Median (50%) : {q50:.4f}")
    print(f"Warning (90%): {q90:.4f}")
    print(f"High (95%)   : {q95:.4f}")
    print(f"Critical(99%): {q99:.4f}")
    print(f"[+] Saved thresholds to: {THRESHOLDS_PATH}\n")