from pathlib import Path
import pandas as pd
import yaml

with open("config/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_file = Path(config["data"]["expression_file"])
meta_file = Path(config["data"]["metadata_file"])

print("Expression file:", expr_file)
print("Exists:", expr_file.exists())

print("Metadata file:", meta_file)
print("Exists:", meta_file.exists())

if expr_file.exists():
    expr_preview = pd.read_csv(expr_file, index_col=0, nrows=5)
    print("Expression preview:", expr_preview.shape)
    print(expr_preview.iloc[:3, :3])

if meta_file.exists():
    meta = pd.read_csv(meta_file, index_col=0)
    print("Metadata:", meta.shape)
    print(meta[["condition", "disease", "GSE", "GPL", "sample_id"]].head())
