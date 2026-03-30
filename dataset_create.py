import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

pos_data = np.load("positive_samples.npz")
neg_data = np.load("negative_samples.npz")

positive_samples = list(zip(pos_data["X"], pos_data["y"]))
negative_samples = list(zip(neg_data["X"], neg_data["y"]))

all_samples = positive_samples + negative_samples
random.shuffle(all_samples)

X = np.array([s[0] for s in all_samples])  # (N, 332)
y = np.array([s[1] for s in all_samples])  # (N,)

# 划分 train/val/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}")  # ~(24000, 332)
print(f"Val:   {X_val.shape}")    # ~(3000, 332)
print(f"Test:  {X_test.shape}")   # ~(6000, 332)

selector = VarianceThreshold(threshold=0.0)  # 去掉方差为0的列
X_train = selector.fit_transform(X_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

print(f"Train: {X_train.shape}")  # ~(24000, 332)
print(f"Val:   {X_val.shape}")    # ~(3000, 332)
print(f"Test:  {X_test.shape}")   # ~(6000, 332)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy",   X_val)
np.save("y_val.npy",   y_val)
np.save("X_test.npy",  X_test)
np.save("y_test.npy",  y_test)