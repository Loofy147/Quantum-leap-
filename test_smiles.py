import os
from cross_domain.domain_adapters import DrugDiscoveryAdapter, load_kaggle_smiles_data

k_smiles = "data/drug_discovery/DDH Data.csv"
if os.path.exists(k_smiles):
    data = load_kaggle_smiles_data(k_smiles)
    print(f"Loaded {len(data['compounds'])} SMILES")
    drug = DrugDiscoveryAdapter()
    res = drug.build_compound_database(drug_data=data)
    print(f"Indexed: {res['compounds_indexed']}, Source: {res['source']}")
else:
    print("CSV not found")
