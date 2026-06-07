import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MarketStressTransformer


X_train = np.load("data/processed/X_train.npy")
Y_train = np.load("data/processed/Y_train.npy")

X_val = np.load("data/processed/X_val.npy")
Y_val = np.load("data/processed/Y_val.npy")

X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()

X_val = torch.from_numpy(X_val).float()
Y_val = torch.from_numpy(Y_val).float()

batch_size = 128

class StressDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

train_dataset = StressDataset(X_train, Y_train)
val_dataset = StressDataset(X_val, Y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

model = MarketStressTransformer()
loss_measure = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5  

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(train_loader):

        prediction = model(x)

        loss = loss_measure(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.6f}")
    
    print(f"\nEpoch {epoch} completed | Avg Loss: {total_loss / len(train_loader):.6f}\n")

