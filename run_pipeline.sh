#!/usr/bin/env bash
set -e  # Stop execution immediately if any command fails

export PYTHONHASHSEED=0

echo "=== 1/9 Running paths.py ==="
python scripts/paths.py

echo "=== 2/9 Running 01_build_sample_metadata.py ==="
python scripts/01_build_sample_metadata.py

echo "=== 3/9 Running 02_map_and_clean_gene_identifiers.py ==="
python scripts/02_map_and_clean_gene_identifiers.py

echo "=== 4/9 Running 03_filter_normalize_log2cpm.py ==="
python scripts/03_filter_normalize_log2cpm.py

echo "=== 5/9 Running 04_tune_random_forest_oob.py ==="
python scripts/04_tune_random_forest_oob.py

echo "=== 6/9 Running 05_select_top_gene_signatures.py ==="
python scripts/05_select_top_gene_signatures.py

echo "=== 7/9 Running 06_run_go_kegg_enrichment.py ==="
python scripts/06_run_go_kegg_enrichment.py

echo "=== 8/9 Running 07_prepare_ppi_cytoscape_inputs.py ==="
python scripts/07_prepare_ppi_cytoscape_inputs.py

echo "=== 9/9 Running 08_string_ppi_hub_analysis.py ==="
python scripts/08_string_ppi_hub_analysis.py

echo "=== Running 09_validate_severity_signature.py ==="
python scripts/09_validate_severity_signature.py

echo "=== Pipeline Completed Successfully ==="