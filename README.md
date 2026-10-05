# Dengue Severity and Warning-Sign Analysis

This repository contains the reproducible analysis pipeline for investigating distinct blood transcriptional programs associated with dengue severity and warning signs of plasma leakage.

## Analysis Axes

- **Severity:** DHF vs DF
- **Warning signs:** D+W vs D-W

## Datasets

- **GSE215835** — severity discovery cohort
- **GSE178240** — warning-sign analysis
- **GSE43777 / GPL570** — independent validation of the severity signature

## Analysis Workflow

- **Step 01 — Metadata preparation**  
  Builds standardized sample metadata.

- **Step 02 — Gene identifier processing**  
  Maps and cleans gene identifiers.

- **Step 03 — Filtering and normalization**  
  Generates filtered and normalized log2-CPM expression matrices.

- **Step 04 — Random Forest OOB tuning**  
  Selects the number of trees for each analysis axis using out-of-bag error.

- **Step 05 — Gene-signature selection and validation**  
  Identifies RF-derived gene signatures, evaluates signature sizes, performs nested cross-validation, and performs held-out **NovaSeq 6000 → HiSeq 2500 validation** for the warning-sign axis.

- **Step 06 — GO-BP and KEGG enrichment**  
  Identifies biological processes and pathways associated with each gene signature.

- **Step 07 — PPI input preparation**  
  Generates STRING-ready gene lists and Cytoscape node annotations.

- **Step 08 — STRING PPI and hub analysis**  
  Constructs PPI networks and ranks candidate hub genes using MCC, Degree, and MNC.

- **Step 09 — Independent severity validation**  
  Validates the fixed severity signature in **GSE43777/GPL570** using direction replication and cross-validated RF AUC.

## Outputs

Main results are stored under:

```text
scripts_outcomes/
```

## Running the Pipeline

Complete setup and execution instructions are available in:

[`RUNNING.md`](RUNNING.md)

## Configuration

Pipeline settings are defined in:

```text
config/paths.yaml
```

## Reproducibility

The pipeline uses fixed random seeds and records relevant analysis settings and provenance information for reproducible execution.
