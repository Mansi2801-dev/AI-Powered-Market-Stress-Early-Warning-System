import pandas as pd
import numpy as np

X = np.load("data/processed/X.npy")
Y = np.load("data/processed/Y.npy")

total_samples = len(X)

train_samples = int(total_samples * 0.70)
validation_samples = int(total_samples * (0.70 + 0.15))

X_train , Y_train = X[:train_samples], Y[:train_samples]
X_val , Y_val = X[train_samples:validation_samples], Y[train_samples:validation_samples]
X_test , Y_test = X[validation_samples: ], Y[validation_samples: ]

print(f"Total samples: {total_samples}")
print("Train")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("\nValidation")
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("\nTest")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/Y_train.npy", Y_train)

np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/Y_val.npy", Y_val)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/Y_test.npy", Y_test)