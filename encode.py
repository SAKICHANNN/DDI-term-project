from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import json

with open("drug_smiles.json", "r") as f:
    drug_smiles = json.load(f)

def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # 无效的SMILES
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)  # 166维的binary向量

# 为每个药生成指纹
drug_fingerprints = {}
for drug_id, smiles in drug_smiles.items():
    fp = smiles_to_maccs(smiles)
    if fp is not None:
        drug_fingerprints[drug_id] = fp

np.savez(
    "drug_fingerprints.npz",
    ids=list(drug_fingerprints.keys()),
    fps=np.array(list(drug_fingerprints.values()))
)

# 验证一下
example_id = list(drug_fingerprints.keys())[0]
print(f"指纹维度: {drug_fingerprints[example_id].shape}")  # (166,)
print(f"指纹示例: {drug_fingerprints[example_id]}")        # [0 0 1 0 1 1 ...]