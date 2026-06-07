import numpy as np

X = np.load("data/processed/X.npy")
Y = np.load("data/processed/Y.npy")


assert len(X) == len(Y), "X and Y mismatch!"

total_samples = len(X)


train_end = int(total_samples * 0.70)
val_end = int(total_samples * 0.85)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

print(f"Total samples: {total_samples}")

print("\nTrain")
print(X_train.shape, Y_train.shape)

print("\nValidation")
print(X_val.shape, Y_val.shape)

print("\nTest")
print(X_test.shape, Y_test.shape)


np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/Y_train.npy", Y_train)

np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/Y_val.npy", Y_val)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/Y_test.npy", Y_test)