from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

x_path = PROCESSED_DIR / "X.npy"
y_path = PROCESSED_DIR / "Y.npy" if (PROCESSED_DIR / "Y.npy").exists() else PROCESSED_DIR / "y.npy"

if not x_path.exists() or not y_path.exists():
    raise FileNotFoundError(f"Missing sequence files in {PROCESSED_DIR}. Run sequence_dataset.py first.")

X = np.load(x_path)
Y = np.load(y_path)

assert len(X) == len(Y), f"Length mismatch: X={len(X)}, Y={len(Y)}"
total_samples = len(X)

train_end = int(total_samples * 0.70)
val_end = int(total_samples * 0.85)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

print(f"[+] Total samples: {total_samples}")
print(f"[+] Train split : X={X_train.shape}, Y={Y_train.shape}")
print(f"[+] Val split   : X={X_val.shape}, Y={Y_val.shape}")
print(f"[+] Test split  : X={X_test.shape}, Y={Y_test.shape}")

np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "Y_train.npy", Y_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "Y_val.npy", Y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "Y_test.npy", Y_test)

print(f"[+] All split files saved to: {PROCESSED_DIR}\n")