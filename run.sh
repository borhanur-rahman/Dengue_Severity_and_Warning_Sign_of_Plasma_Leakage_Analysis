#!/bin/bash
set -e
cd /code
export PYTHONHASHSEED=0
python scripts/01_build_sample_metadata.py
python scripts/02_map_and_clean_gene_identifiers.py
python scripts/03_filter_normalize_log2cpm.py
python scripts/04_tune_random_forest_oob.py
python scripts/05_select_top_gene_signatures.py
python scripts/06_run_go_kegg_enrichment.py
python scripts/07_prepare_ppi_cytoscape_inputs.py
python scripts/07_string_ppi_hub_analysis.py
python scripts/08_validate_severity_signature.py
cp -r scripts_outcomes/runs /results/