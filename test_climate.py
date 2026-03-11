import os
from cross_domain.domain_adapters import load_kaggle_climate_data, ClimateAdapter

path = "./data/climate/climate_change_indicators.csv"
if os.path.exists(path):
    series = load_kaggle_climate_data(path)
    print(f"Loaded series shape: {series.shape}")
    cli = ClimateAdapter()
    res = cli.analyze(series)
    print(f"Analysis result: {res['source'] if 'source' in res else 'No source'}")
    print(f"Steps: {res['n_steps']}")
else:
    print("CSV not found")
