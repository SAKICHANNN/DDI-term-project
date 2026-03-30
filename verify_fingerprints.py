import numpy as np

data = np.load("drug_fingerprints.npz", allow_pickle=True)
drug_fingerprints = dict(zip(data["ids"], data["fps"]))

# 验证一下
print(f"加载了{len(drug_fingerprints)}个药物的指纹")
print(f"指纹维度: {list(drug_fingerprints.values())[0].shape}")  # (166,)