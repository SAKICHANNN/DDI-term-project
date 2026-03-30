import pandas as pd
import numpy as np

data = np.load("drug_fingerprints.npz", allow_pickle=True)
drug_fingerprints = dict(zip(data["ids"], data["fps"]))

df = pd.read_csv("ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])

positive_samples = []

for _, row in df.iterrows():
    d1, d2 = row["drug1"], row["drug2"]
    
    # 过滤掉没有SMILES的药
    if d1 not in drug_fingerprints or d2 not in drug_fingerprints:
        continue
    
    fp1 = drug_fingerprints[d1]
    fp2 = drug_fingerprints[d2]
    
    # 拼接两个指纹
    combined = np.concatenate([fp1, fp2])  # 166 + 166 = 332维
    positive_samples.append((combined, 1))

print(f"正例数量: {len(positive_samples)}")  # ~15,000左右

X_pos = np.array([s[0] for s in positive_samples])
y_pos = np.array([s[1] for s in positive_samples])
np.savez("positive_samples.npz", X=X_pos, y=y_pos)