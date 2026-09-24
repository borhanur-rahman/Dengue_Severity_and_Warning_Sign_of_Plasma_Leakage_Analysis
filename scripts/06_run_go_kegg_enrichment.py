#!/usr/bin/env python3
"""Step 05 — enrichment (GO-BP + KEGG) on each axis's top-50 RF genes.

Input : v2/runs/<RUN_ID>/rf/{leakage,severity}_top50_features.tsv (from step 04)
Output: v2/runs/<RUN_ID>/enrichment/
Method == your original enrichment script (Enrichr, GO-BP + KEGG, padj<0.05).
Reproducibility: Enrichr library versions recorded to _provenance.txt.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import gseapy as gp

from paths import CFG, seed_everything, resolve, read_dir, run_dir

warnings.filterwarnings("ignore")
seed_everything()

_en = CFG["enrichment"]
GO_LIB = _en["go_library"]
KEGG_LIB = _en["kegg_library"]
PADJ = float(_en["padj"])
TOPN = int(_en["top_n"])
AXES = list(CFG["ppi"]["axes"])          # [leakage, severity]

RUN_ID = None
RF_DIR: Path = None
OUT: Path = None

def get_top50(axis):
    f = RF_DIR / f"{axis}_top{TOPN}_features.tsv"
    if not f.exists():
        raise SystemExit(f"[{axis}] {f} not found — run step 04 first.")
    g = pd.read_csv(f, sep="\t")["gene"].astype(str).tolist()[:TOPN]
    print(f"[{axis}] loaded {len(g)} genes from {f}")
    return g


def clean_genes(genes):
    out = []
    for g in genes:
        g = str(g).strip()
        if not g or g.upper().startswith("ENSG"):
            continue
        out.append(g)
    return out


def enrich(genes, library, tag):
    genes = clean_genes(genes)
    try:
        res = gp.enrichr(gene_list=genes, gene_sets=library, organism="human",
                         outdir=None, no_plot=True).results
    except Exception as e:
        print(f"  [{tag}] enrichr failed: {e}")
        return pd.DataFrame()
    if res is None or res.empty:
        return pd.DataFrame()
    res = res.rename(columns={"Adjusted P-value": "padj", "P-value": "pvalue",
                              "Term": "term", "Genes": "genes", "Overlap": "overlap"})
    sig = res[res["padj"] < PADJ].copy()
    keep = [c for c in ["term", "overlap", "pvalue", "padj", "Combined Score", "genes"] if c in sig.columns]
    sig = sig[keep].sort_values("padj").reset_index(drop=True)
    sig.insert(0, "library", "GO_BP" if "GO_" in library else "KEGG")
    return sig


def genes_from_terms(df):
    s = set()
    for g in df.get("genes", pd.Series(dtype=str)).dropna():
        s |= {x.strip() for x in str(g).split(";") if x.strip()}
    return s


def run_axis(axis):
    print("\n" + "=" * 70 + f"\nAXIS: {axis}  (top-{TOPN} genes)\n" + "=" * 70)
    genes = get_top50(axis)
    go = enrich(genes, GO_LIB, f"{axis}/GO-BP")
    kegg = enrich(genes, KEGG_LIB, f"{axis}/KEGG")

    print(f"[{axis}] GO-BP significant terms (padj<{PADJ}): {len(go)}")
    if len(go):
        print(go.head(10)[["term", "overlap", "padj"]].to_string(index=False))
    print(f"\n[{axis}] KEGG significant terms (padj<{PADJ}): {len(kegg)}")
    if len(kegg):
        print(kegg.head(10)[["term", "overlap", "padj"]].to_string(index=False))

    go.to_csv(OUT / f"{axis}_GO_BP_significant.tsv", sep="\t", index=False)
    kegg.to_csv(OUT / f"{axis}_KEGG_significant.tsv", sep="\t", index=False)

    union = pd.concat([go, kegg], ignore_index=True).sort_values("padj").reset_index(drop=True)
    union.to_csv(OUT / f"{axis}_GOBP_KEGG_union_terms.tsv", sep="\t", index=False)
    axis_genes = genes_from_terms(union)
    pd.DataFrame({"gene": sorted(axis_genes)}).to_csv(OUT / f"{axis}_enriched_genes_union.tsv",
                                                      sep="\t", index=False)
    print(f"\n[{axis}] WITHIN-AXIS union (GO-BP u KEGG): {len(union)} terms, "
          f"{len(axis_genes)} associated genes")
    return {"axis": axis, "go": go, "kegg": kegg, "union_terms": union, "union_genes": axis_genes}


def main():
    global RF_DIR, OUT
    rid = resolve(RUN_ID)
    RF_DIR = read_dir(rid, "rf")
    OUT = run_dir(rid, "enrichment", config=_en | {"axes": AXES})
    (OUT / "_provenance.txt").write_text(
        f"gseapy: {getattr(gp,'__version__','?')}\nGO: {GO_LIB}\nKEGG: {KEGG_LIB}\n"
        f"padj: {PADJ}\ntop_n: {TOPN}\nCite these exact library names.\n")
    print(f"  reading: {RF_DIR}\n  writing: {OUT}")
    leak = run_axis("leakage")
    sev = run_axis("severity")

    def tag(df, axis):
        d = df.copy(); d.insert(0, "axis", axis); return d
    cross = pd.concat([tag(sev["union_terms"], "severity"),
                       tag(leak["union_terms"], "leakage")], ignore_index=True)
    cross.to_csv(OUT / "cross_axis_union_terms.tsv", sep="\t", index=False)

    sev_terms = set(sev["union_terms"]["term"]); leak_terms = set(leak["union_terms"]["term"])
    shared = sorted(sev_terms & leak_terms)
    gene_union = sev["union_genes"] | leak["union_genes"]
    gene_shared = sev["union_genes"] & leak["union_genes"]
    pd.DataFrame({"gene": sorted(gene_union)}).to_csv(OUT / "cross_axis_gene_union.tsv", sep="\t", index=False)

    print("\n" + "=" * 70 + "\nCROSS-AXIS SUMMARY\n" + "=" * 70)
    print(f"severity: {len(sev_terms)} sig terms | leakage: {len(leak_terms)} sig terms")
    print(f"shared terms (severity n leakage): {len(shared)}")
    if shared:
        print("  " + "; ".join(shared[:10]) + (" ..." if len(shared) > 10 else ""))
    print(f"term union (severity u leakage): {len(sev_terms | leak_terms)}")
    print(f"enriched-gene union: {len(gene_union)} | enriched-gene shared: {len(gene_shared)}")
    if gene_shared:
        print("  shared genes: " + ", ".join(sorted(gene_shared)))

    pd.DataFrame([{
        "severity_go_terms": len(sev["go"]), "severity_kegg_terms": len(sev["kegg"]),
        "leakage_go_terms": len(leak["go"]), "leakage_kegg_terms": len(leak["kegg"]),
        "severity_union_terms": len(sev_terms), "leakage_union_terms": len(leak_terms),
        "shared_terms": len(shared), "term_union": len(sev_terms | leak_terms),
        "gene_union": len(gene_union), "gene_shared": len(gene_shared)}]).to_csv(
        OUT / "enrichment_summary.tsv", sep="\t", index=False)

    print(f"\n[SAVED] -> {OUT}")
    print("[DONE]")


if __name__ == "__main__":
    main()