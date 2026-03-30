import pandas as pd
import requests
import time

df = pd.read_csv(r"ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])

def get_smiles_from_pubchem(drugbank_id):
    # PubChem可以通过DrugBank ID查询SMILES
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{drugbank_id}/property/IsomericSMILES,CanonicalSMILES,SMILES/JSON"
    response = requests.get(url)
    time.sleep(0.2)  # 避免请求太快被限速
    
    if response.status_code == 200:
        data = response.json()
        props = data["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("SMILES")
    return None

# 收集所有唯一的drug ID
all_drug_ids = set(df["drug1"].tolist() + df["drug2"].tolist())

# 批量查询SMILES
drug_smiles = {}
for drug_id in all_drug_ids:
    smiles = get_smiles_from_pubchem(drug_id)
    if smiles:
        drug_smiles[drug_id] = smiles
    print(f"{drug_id}: {smiles}")

# 保存下来，避免重复请求
import json
with open("drug_smiles.json", "w") as f:
    json.dump(drug_smiles, f)