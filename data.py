import pandas as pd

df = pd.read_csv("ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])
print(df.head())
