import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1. Path Setup
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH / "src"))

from model import MarketStressTransformer

# 2. Hardware & Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Active Device: {device.type.upper()}")

if device.type == "cpu":
    # Maximize CPU core utilization on Windows
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"[+] CPU multi-threading enabled: {torch.get_num_threads()} threads")

DATA_PATH = PROJECT_PATH / "data" / "processed"
MODEL_PATH = PROJECT_PATH / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 3. Load Datasets
print("[+] Loading preprocessed dataset arrays...")
X_train = np.load(DATA_PATH / "X_train.npy")
Y_train = np.load(DATA_PATH / "Y_train.npy")
X_val = np.load(DATA_PATH / "X_val.npy")
Y_val = np.load(DATA_PATH / "Y_val.npy")

# ==============================================================================
# SPEED OPTIMIZATION FOR CPU TRAINING
# Since 60-step rolling windows have 98% identical data, a stride of 2 cuts 
# training time by 50% with negligible impact on accuracy.
# (Set SAMPLE_STRIDE = 1 if you want to use 100% of all possible windows)
# ==============================================================================
SAMPLE_STRIDE = 2
if SAMPLE_STRIDE > 1:
    X_train = X_train[::SAMPLE_STRIDE]
    Y_train = Y_train[::SAMPLE_STRIDE]
    X_val = X_val[::SAMPLE_STRIDE]
    Y_val = Y_val[::SAMPLE_STRIDE]
    print(f"[+] Applied stride={SAMPLE_STRIDE} -> Accelerated Train samples: {len(X_train):,}")

# 4. Zero-Copy PyTorch Dataset
class StressDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # from_numpy avoids duplicating gigabytes in memory
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# On CPU, a batch size of 256 vectorizes significantly better than 128
batch_size = 256 if device.type == "cpu" else 128

train_loader = DataLoader(
    StressDataset(X_train, Y_train),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    StressDataset(X_val, Y_val),
    batch_size=batch_size,
    shuffle=False
)

# 5. Model, Loss, Optimizer
model = MarketStressTransformer().to(device)
loss_fn = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=1, factor=0.5
)

# On CPU, 5 epochs is sufficient with normalized features
epochs = 5
best_val_loss = float("inf")
save_dest = MODEL_PATH / "stress_model.pth"

print(f"\n[+] Total Batches per Epoch: {len(train_loader)}")
print(f"[+] Starting training for {epochs} epochs...\n")

# 6. Training Loop with Live Progress Bar
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Live progress bar using tqdm
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]",
        leave=True,
        dynamic_ncols=True
    )

    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        # Update progress bar with live loss in real time
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / len(train_loader)

    # 7. Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += loss_fn(pred, y).item()

    epoch_val_loss = val_loss / len(val_loader)
    scheduler.step(epoch_val_loss)

    print(
        f"  --> Epoch {epoch+1:02d} Summary: "
        f"Train Loss = {epoch_train_loss:.5f} | Val Loss = {epoch_val_loss:.5f}"
    )

    # Save Best Model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "d_model": 64,
            "n_layers": 2,
            "best_val_loss": best_val_loss,
            "epoch": epoch + 1
        }
        torch.save(checkpoint, save_dest)
        print(f"  --> [★] Best model saved! (Val Loss: {best_val_loss:.5f})\n")
    else:
        print()

print(f"[+] Training complete. Best model checkpoint saved to: {save_dest}")