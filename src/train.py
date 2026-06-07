import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from model import MarketStressTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_PATH, "Y_train.npy"))

X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_PATH, "Y_val.npy"))

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)


class StressDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

batch_size = 128

train_loader = DataLoader(
    StressDataset(X_train, Y_train),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    StressDataset(X_val, Y_val),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)


model = MarketStressTransformer().to(device)

loss_fn = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.5
)


epochs = 15

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss += loss.item()

        if i % 100 == 0:
            print(f"[Epoch {epoch}] Batch {i} Loss: {loss.item():.6f}")


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"\nEpoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}\n")


MODEL_PATH = os.path.join(PROJECT_PATH, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "stress_model.pth"))

print("Model saved successfully.")