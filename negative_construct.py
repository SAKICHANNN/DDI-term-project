import random
import pandas as pd
import numpy as np

data = np.load("drug_fingerprints.npz", allow_pickle=True)
drug_fingerprints = dict(zip(data["ids"], data["fps"]))

pos_data = np.load("positive_samples.npz")
X_pos, y_pos = pos_data["X"], pos_data["y"]
positive_samples = list(zip(X_pos, y_pos))

df = pd.read_csv(r"ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])

drug_ids = list(drug_fingerprints.keys())

# 把正例药对存成set，用于快速查找
positive_pairs = set()
for _, row in df.iterrows():
    positive_pairs.add((row["drug1"], row["drug2"]))
    positive_pairs.add((row["drug2"], row["drug1"]))  # 双向都记录

negative_samples = []
target_count = len(positive_samples)  # 1:1比例

while len(negative_samples) < target_count:
    d1 = random.choice(drug_ids)
    d2 = random.choice(drug_ids)
    
    if d1 == d2:
        continue
    if (d1, d2) in positive_pairs:
        continue  # 跳过已知正例
    
    fp1 = drug_fingerprints[d1]
    fp2 = drug_fingerprints[d2]
    combined = np.concatenate([fp1, fp2])
    negative_samples.append((combined, 0))

X_pos = np.array([s[0] for s in negative_samples])
y_pos = np.array([s[1] for s in negative_samples])
np.savez("negative_samples.npz", X=X_pos, y=y_pos)

print(f"负例数量: {len(negative_samples)}")