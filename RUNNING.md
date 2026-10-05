# Pipeline Execution Guide

## 1. Clone the Repository

```bash
git clone https://github.com/borhanur-rahman/Dengue_Severity_and_Warning_Sign_of_Plasma_Leakage_Analysis.git
```

```bash
cd Dengue_Sevirity_and_Warning_Sign_of_Plasma_Leakage_Analysis
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Full Pipeline

```bash
chmod +x run_pipeline.sh
```

```bash
./run_pipeline.sh
```

---

# Alternative: Run Each Step Manually

First activate the environment:

```bash
source .venv/bin/activate
```

### Step 01 — Build Metadata

```bash
PYTHONHASHSEED=0 python scripts/01_build_sample_metadata.py
```

### Step 02 — Map and Clean Gene IDs

```bash
PYTHONHASHSEED=0 python scripts/02_map_and_clean_gene_identifiers.py
```

### Step 03 — Filter and Normalize Data

```bash
PYTHONHASHSEED=0 python scripts/03_filter_normalize_log2cpm.py
```

### Step 04 — Tune Random Forest

```bash
PYTHONHASHSEED=0 python scripts/04_tune_random_forest_oob.py
```

### Step 05 — Select Top Gene Signatures

```bash
PYTHONHASHSEED=0 python scripts/05_select_top_gene_signatures.py
```

### Step 06 — GO/KEGG Enrichment

```bash
PYTHONHASHSEED=0 python scripts/06_run_go_kegg_enrichment.py
```

### Step 07 — Prepare PPI Inputs

```bash
PYTHONHASHSEED=0 python scripts/07_prepare_ppi_cytoscape_inputs.py
```

### Step 08 — STRING PPI and Hub Analysis

```bash
PYTHONHASHSEED=0 python scripts/08_string_ppi_hub_analysis.py
```

### Step 09 — Validate Severity Signature

```bash
PYTHONHASHSEED=0 python scripts/09_validate_severity_signature.py
```

---

## Main Output Folder

All generated results are stored in:

```text
scripts_outcomes/
```

Important result folders include:

```text
scripts_outcomes/qc/
scripts_outcomes/runs/results/rf/
scripts_outcomes/runs/results/enrichment/
scripts_outcomes/runs/results/ppi/
scripts_outcomes/runs/results/ppi_hub_analysis/
scripts_outcomes/runs/results/external_validation/
```

---

## Configuration

Main configuration file:

```text
config/paths.yaml
```

Shared path/configuration helper:

```text
scripts/paths.py
```

---

## Reproducibility

Run Python scripts with:

```bash
PYTHONHASHSEED=0
```

The main random seed is also defined in `config/paths.yaml`.

---

## Pipeline Order

```text
Step 01 → Metadata
Step 02 → Gene ID Cleaning
Step 03 → Normalization
Step 04 → RF Tuning
Step 05 → Gene Signature Selection
Step 06 → GO/KEGG Enrichment
Step 07 → PPI Input Preparation
Step 08 → STRING PPI + Hub Analysis
Step 09 → External Severity Validation
```
