#!/usr/bin/env python3
"""
Step 06 — GO Biological Process and KEGG enrichment analysis.

Purpose
-------
Perform functional enrichment analysis independently for the RF-derived
top-gene signature from each dengue analysis axis:

    1. warning_sign
    2. severity

Input
-----
Top-ranked RF genes produced by Step 05:

scripts_outcomes/runs/<RUN_ID>/rf/
    warning_sign_top50_features.tsv
    severity_top50_features.tsv

The number of input genes is controlled by:

    enrichment.top_n

in config/paths.yaml.

Output
------
Results are written to:

scripts_outcomes/runs/<RUN_ID>/enrichment/

For each axis, the script generates:

    <axis>_GO_BP_significant.tsv
    <axis>_KEGG_significant.tsv
    <axis>_GOBP_KEGG_union_terms.tsv
    <axis>_enriched_genes_union.tsv

Cross-axis outputs:

    cross_axis_union_terms.tsv
    cross_axis_gene_union.tsv
    enrichment_summary.tsv

A provenance record is also written to:

    _provenance.txt

Method
------
Enrichment is performed using GSEApy's Enrichr interface.

The enrichment libraries are specified in config/paths.yaml. The current
configuration uses:

    GO_Biological_Process_2021
    KEGG_2021_Human

Terms are considered significant when:

    adjusted P-value < enrichment.padj

The analysis is performed independently for the warning-sign and severity
axes. Significant GO-BP and KEGG terms are subsequently combined within
each axis. Genes contributing to the significant terms are also collected
for downstream STRING/PPI analysis.

Cross-axis analysis
-------------------
The script reports:

    - number of significant GO-BP terms per axis
    - number of significant KEGG terms per axis
    - total significant-term union per axis
    - shared enriched terms between axes
    - cross-axis term union
    - enriched-gene union
    - enriched genes shared between axes

Reproducibility
---------------
The following information is recorded in _provenance.txt:

    - GSEApy version
    - exact Enrichr library names
    - adjusted-P-value threshold
    - input signature size
    - analyzed axes

Important:
Enrichr is an online resource. Recording the exact library names and
software version improves reproducibility, but remote library contents
may be updated independently by Enrichr over time.

No machine-learning model fitting or feature selection is performed in
this step. The gene signatures are fixed outputs from Step 05.
"""

from pathlib import Path
import warnings

import pandas as pd
import gseapy as gp

from paths import (
    CFG,
    seed_everything,
    resolve,
    read_dir,
    run_dir,
)


# =====================================================================
# Global setup
# =====================================================================

warnings.filterwarnings("ignore")

seed_everything()


# =====================================================================
# Configuration
# =====================================================================

_en = CFG["enrichment"]

GO_LIB = _en["go_library"]
KEGG_LIB = _en["kegg_library"]

PADJ = float(_en["padj"])
TOPN = int(_en["top_n"])

# Axes are kept in the configured PPI order because the enrichment
# outputs are subsequently used by the PPI workflow.
#
# Current configuration:
#   warning_sign
#   severity
AXES = list(CFG["ppi"]["axes"])


RUN_ID = None

RF_DIR: Path | None = None
OUT: Path | None = None


# =====================================================================
# Input gene loading
# =====================================================================

def get_top50(axis):
    """
    Load the fixed top-N RF genes produced by Step 05 for one axis.

    Parameters
    ----------
    axis : str
        Analysis axis, e.g. "warning_sign" or "severity".

    Returns
    -------
    list[str]
        Top-ranked RF genes, limited to TOPN.
    """

    f = RF_DIR / f"{axis}_top{TOPN}_features.tsv"

    if not f.exists():
        raise SystemExit(
            f"[{axis}] {f} not found — run Step 05 first."
        )

    df = pd.read_csv(
        f,
        sep="\t",
    )

    if "gene" not in df.columns:
        raise SystemExit(
            f"[{axis}] required column 'gene' not found in {f}"
        )

    genes = (
        df["gene"]
        .astype(str)
        .tolist()[:TOPN]
    )

    print(
        f"[{axis}] loaded {len(genes)} genes from {f}"
    )

    return genes


# =====================================================================
# Gene-symbol cleanup
# =====================================================================

def clean_genes(genes):
    """
    Prepare gene symbols for Enrichr submission.

    Blank entries and Ensembl-style identifiers beginning with ENSG are
    excluded. Step 05 is expected to provide gene symbols, so this
    function serves as a defensive cleanup step.

    Parameters
    ----------
    genes : iterable
        Input gene identifiers.

    Returns
    -------
    list[str]
        Cleaned gene-symbol list.
    """

    cleaned = []

    for gene in genes:

        gene = str(gene).strip()

        if not gene:
            continue

        if gene.upper().startswith("ENSG"):
            continue

        cleaned.append(gene)

    return cleaned


# =====================================================================
# Enrichment
# =====================================================================

def enrich(genes, library, tag):
    """
    Run Enrichr enrichment for one gene list and one library.

    Significant terms are retained using:

        adjusted P-value < PADJ

    Parameters
    ----------
    genes : list[str]
        Input gene symbols.

    library : str
        Enrichr library name.

    tag : str
        Label used in console messages.

    Returns
    -------
    pandas.DataFrame
        Significant enrichment results.
    """

    genes = clean_genes(
        genes
    )

    if not genes:
        print(
            f"  [{tag}] no valid genes available for enrichment"
        )

        return pd.DataFrame()

    try:

        result = gp.enrichr(
            gene_list=genes,
            gene_sets=library,
            organism="human",
            outdir=None,
            no_plot=True,
        ).results

    except Exception as exc:

        print(
            f"  [{tag}] Enrichr failed: {exc}"
        )

        return pd.DataFrame()

    if result is None or result.empty:
        return pd.DataFrame()

    result = result.rename(
        columns={
            "Adjusted P-value": "padj",
            "P-value": "pvalue",
            "Term": "term",
            "Genes": "genes",
            "Overlap": "overlap",
        }
    )

    significant = result[
        result["padj"] < PADJ
    ].copy()

    keep = [
        column
        for column in [
            "term",
            "overlap",
            "pvalue",
            "padj",
            "Combined Score",
            "genes",
        ]
        if column in significant.columns
    ]

    significant = (
        significant[keep]
        .sort_values("padj")
        .reset_index(drop=True)
    )

    if "GO_" in library:
        library_label = "GO_BP"
    else:
        library_label = "KEGG"

    significant.insert(
        0,
        "library",
        library_label,
    )

    return significant


# =====================================================================
# Genes represented by significant enriched terms
# =====================================================================

def genes_from_terms(df):
    """
    Extract the union of genes represented in significant enrichment terms.

    Enrichr stores term-associated genes as semicolon-separated symbols.

    Parameters
    ----------
    df : pandas.DataFrame
        Significant enrichment table.

    Returns
    -------
    set[str]
        Unique genes represented across the terms.
    """

    genes = set()

    if df.empty or "genes" not in df.columns:
        return genes

    for value in df["genes"].dropna():

        genes |= {
            gene.strip()
            for gene in str(value).split(";")
            if gene.strip()
        }

    return genes


# =====================================================================
# Per-axis enrichment
# =====================================================================

def run_axis(axis):
    """
    Run GO-BP and KEGG enrichment independently for one analysis axis.

    The significant terms are saved separately and then combined into a
    within-axis GO-BP/KEGG union. Genes represented by the significant
    terms are exported for downstream PPI analysis.
    """

    print(
        "\n"
        + "=" * 70
        + f"\nAXIS: {axis}  (top-{TOPN} genes)\n"
        + "=" * 70
    )

    genes = get_top50(
        axis
    )

    # -----------------------------------------------------------------
    # GO Biological Process
    # -----------------------------------------------------------------

    go = enrich(
        genes,
        GO_LIB,
        f"{axis}/GO-BP",
    )

    # -----------------------------------------------------------------
    # KEGG
    # -----------------------------------------------------------------

    kegg = enrich(
        genes,
        KEGG_LIB,
        f"{axis}/KEGG",
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print(
        f"[{axis}] GO-BP significant terms "
        f"(padj<{PADJ}): {len(go)}"
    )

    if len(go):

        print(
            go.head(10)[
                [
                    "term",
                    "overlap",
                    "padj",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        f"\n[{axis}] KEGG significant terms "
        f"(padj<{PADJ}): {len(kegg)}"
    )

    if len(kegg):

        print(
            kegg.head(10)[
                [
                    "term",
                    "overlap",
                    "padj",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------------------
    # Save library-specific results
    # -----------------------------------------------------------------

    go.to_csv(
        OUT / f"{axis}_GO_BP_significant.tsv",
        sep="\t",
        index=False,
    )

    kegg.to_csv(
        OUT / f"{axis}_KEGG_significant.tsv",
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Within-axis significant-term union
    # -----------------------------------------------------------------

    union = pd.concat(
        [
            go,
            kegg,
        ],
        ignore_index=True,
    )

    if not union.empty and "padj" in union.columns:

        union = (
            union
            .sort_values("padj")
            .reset_index(drop=True)
        )

    union.to_csv(
        OUT / f"{axis}_GOBP_KEGG_union_terms.tsv",
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Genes represented by significant enriched terms
    # -----------------------------------------------------------------

    axis_genes = genes_from_terms(
        union
    )

    pd.DataFrame(
        {
            "gene": sorted(
                axis_genes
            )
        }
    ).to_csv(
        OUT / f"{axis}_enriched_genes_union.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"\n[{axis}] WITHIN-AXIS union "
        f"(GO-BP ∪ KEGG): "
        f"{len(union)} terms, "
        f"{len(axis_genes)} associated genes"
    )

    return {
        "axis": axis,
        "go": go,
        "kegg": kegg,
        "union_terms": union,
        "union_genes": axis_genes,
    }


# =====================================================================
# Main workflow
# =====================================================================

def main():

    global RF_DIR
    global OUT

    # -----------------------------------------------------------------
    # Resolve the current pipeline run
    # -----------------------------------------------------------------

    rid = resolve(
        RUN_ID
    )

    RF_DIR = read_dir(
        rid,
        "rf",
    )

    OUT = run_dir(
        rid,
        "enrichment",
        config={
            **_en,
            "axes": AXES,
        },
    )

    # -----------------------------------------------------------------
    # Reproducibility / provenance record
    # -----------------------------------------------------------------

    provenance = (
        f"gseapy_version: "
        f"{getattr(gp, '__version__', '?')}\n"
        f"GO_library: {GO_LIB}\n"
        f"KEGG_library: {KEGG_LIB}\n"
        f"adjusted_p_threshold: {PADJ}\n"
        f"top_n: {TOPN}\n"
        f"axes: {', '.join(AXES)}\n"
        "\n"
        "The gene signatures analyzed here are fixed outputs from Step 05.\n"
        "No feature selection or RF model fitting is performed in Step 06.\n"
        "\n"
        "Note: Enrichr is an online resource. The exact library names and "
        "GSEApy version are recorded here, but remote library contents may "
        "change independently over time.\n"
    )

    (
        OUT / "_provenance.txt"
    ).write_text(
        provenance
    )

    print(
        f"  reading: {RF_DIR}\n"
        f"  writing: {OUT}"
    )

    # -----------------------------------------------------------------
    # Run enrichment independently for both axes
    # -----------------------------------------------------------------

    warn = run_axis(
        "warning_sign"
    )

    sev = run_axis(
        "severity"
    )

    # -----------------------------------------------------------------
    # Helper for cross-axis output
    # -----------------------------------------------------------------

    def tag(df, axis):

        tagged = df.copy()

        tagged.insert(
            0,
            "axis",
            axis,
        )

        return tagged

    # -----------------------------------------------------------------
    # Combine all significant terms across axes
    # -----------------------------------------------------------------

    cross = pd.concat(
        [
            tag(
                sev["union_terms"],
                "severity",
            ),

            tag(
                warn["union_terms"],
                "warning_sign",
            ),
        ],
        ignore_index=True,
    )

    cross.to_csv(
        OUT / "cross_axis_union_terms.tsv",
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Cross-axis term overlap
    # -----------------------------------------------------------------

    sev_terms = set(
        sev["union_terms"]["term"]
    )

    warn_terms = set(
        warn["union_terms"]["term"]
    )

    shared_terms = sorted(
        sev_terms
        & warn_terms
    )

    term_union = (
        sev_terms
        | warn_terms
    )

    # -----------------------------------------------------------------
    # Cross-axis enriched-gene overlap
    # -----------------------------------------------------------------

    gene_union = (
        sev["union_genes"]
        | warn["union_genes"]
    )

    gene_shared = (
        sev["union_genes"]
        & warn["union_genes"]
    )

    pd.DataFrame(
        {
            "gene": sorted(
                gene_union
            )
        }
    ).to_csv(
        OUT / "cross_axis_gene_union.tsv",
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
        + "\nCROSS-AXIS SUMMARY\n"
        + "=" * 70
    )

    print(
        f"severity: "
        f"{len(sev_terms)} sig terms | "
        f"warning_sign: "
        f"{len(warn_terms)} sig terms"
    )

    print(
        "shared terms "
        "(severity ∩ warning_sign): "
        f"{len(shared_terms)}"
    )

    if shared_terms:

        print(
            "  "
            + "; ".join(
                shared_terms[:10]
            )
            + (
                " ..."
                if len(shared_terms) > 10
                else ""
            )
        )

    print(
        "term union "
        "(severity ∪ warning_sign): "
        f"{len(term_union)}"
    )

    print(
        f"enriched-gene union: "
        f"{len(gene_union)} | "
        f"enriched-gene shared: "
        f"{len(gene_shared)}"
    )

    if gene_shared:

        print(
            "  shared genes: "
            + ", ".join(
                sorted(
                    gene_shared
                )
            )
        )

    # -----------------------------------------------------------------
    # Save compact summary table
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "severity_go_terms":
                    len(sev["go"]),

                "severity_kegg_terms":
                    len(sev["kegg"]),

                "warning_sign_go_terms":
                    len(warn["go"]),

                "warning_sign_kegg_terms":
                    len(warn["kegg"]),

                "severity_union_terms":
                    len(sev_terms),

                "warning_sign_union_terms":
                    len(warn_terms),

                "shared_terms":
                    len(shared_terms),

                "term_union":
                    len(term_union),

                "gene_union":
                    len(gene_union),

                "gene_shared":
                    len(gene_shared),
            }
        ]
    )

    summary.to_csv(
        OUT / "enrichment_summary.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"\n[SAVED] -> {OUT}"
    )

    print(
        "[DONE]"
    )


if __name__ == "__main__":
    main()