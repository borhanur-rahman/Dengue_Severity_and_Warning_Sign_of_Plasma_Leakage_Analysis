#!/usr/bin/env python3
"""Step 06 — export per-axis enriched-gene unions for STRING -> Cytoscape -> cytoHubba.

Input : v2/runs/<RUN_ID>/enrichment/{axis}_enriched_genes_union.tsv (from step 05)
Output: v2/runs/<RUN_ID>/ppi/
Method unchanged. STRING/cytoHubba performed manually; HOWTO records the exact
STRING confidence and cytoHubba method for reproducibility.
"""
from pathlib import Path

import pandas as pd

from paths import CFG, seed_everything, resolve, read_dir, run_dir

seed_everything()

AXES = list(CFG["ppi"]["axes"])
STRING_MIN_SCORE = float(CFG["ppi"]["string_min_score"])
CYTOHUBBA_METHOD = CFG["ppi"]["cytohubba_method"]

RUN_ID = None
ENR: Path = None
OUT: Path = None

def genes_from_terms(df):
    s = set()
    for g in df.get("genes", pd.Series(dtype=str)).dropna():
        s |= {x.strip() for x in str(g).split(";") if x.strip()}
    return s


def load_axis_union(axis):
    """Prefer the ready-made per-axis enriched-gene union; else rebuild from GO/KEGG tables."""
    ready = ENR / f"{axis}_enriched_genes_union.tsv"
    if ready.exists():
        genes = pd.read_csv(ready, sep="\t")["gene"].astype(str).tolist()
        src = "enriched_genes_union.tsv"
    else:
        go_f = ENR / f"{axis}_GO_BP_significant.tsv"
        kegg_f = ENR / f"{axis}_KEGG_significant.tsv"
        go = pd.read_csv(go_f, sep="\t") if go_f.exists() else pd.DataFrame()
        kegg = pd.read_csv(kegg_f, sep="\t") if kegg_f.exists() else pd.DataFrame()
        genes = sorted(genes_from_terms(go) | genes_from_terms(kegg))
        src = "GO_BP + KEGG significant tables"
    genes = sorted({g for g in genes if g and not str(g).upper().startswith("ENSG")})
    return genes, src


def per_gene_membership(axis, genes):
    """Annotate each gene with which library/terms it came from (handy in Cytoscape)."""
    go_f = ENR / f"{axis}_GO_BP_significant.tsv"; kegg_f = ENR / f"{axis}_KEGG_significant.tsv"
    go = pd.read_csv(go_f, sep="\t") if go_f.exists() else pd.DataFrame()
    kegg = pd.read_csv(kegg_f, sep="\t") if kegg_f.exists() else pd.DataFrame()

    def terms_for(df, gene):
        hits = []
        for _, r in df.iterrows():
            mem = {x.strip() for x in str(r.get("genes", "")).split(";") if x.strip()}
            if gene in mem:
                hits.append(str(r.get("term", "")))
        return hits

    rows = []
    for g in genes:
        gt = terms_for(go, g); kt = terms_for(kegg, g)
        rows.append({"gene": g, "in_GO_BP": bool(gt), "in_KEGG": bool(kt),
                     "n_GO_terms": len(gt), "n_KEGG_terms": len(kt),
                     "GO_terms": " | ".join(gt), "KEGG_terms": " | ".join(kt)})
    return pd.DataFrame(rows)


def main():
    global ENR, OUT
    rid = resolve(RUN_ID)
    ENR = read_dir(rid, "enrichment")
    OUT = run_dir(rid, "ppi", config=CFG["ppi"])
    print(f"  reading: {ENR}\n  writing: {OUT}")
    print("=" * 70 + "\nPPI INPUT EXPORT (STRING -> Cytoscape -> cytoHubba)\n" + "=" * 70)
    for axis in AXES:
        genes, src = load_axis_union(axis)
        if not genes:
            print(f"\n[{axis}] NO enriched genes found (source: {src}). "
                  f"Run top50_enrichment.py first, or the axis had no significant terms.")
            continue

        plain = OUT / f"{axis}_ppi_genes.txt"
        plain.write_text("\n".join(genes) + "\n")
        (OUT / f"{axis}_ppi_genes_oneline.txt").write_text(" ".join(genes) + "\n")

        ann = per_gene_membership(axis, genes)
        ann.to_csv(OUT / f"{axis}_node_attributes.tsv", sep="\t", index=False)

        print(f"\n[{axis}] {len(genes)} enriched genes (from {src})")
        print(f"  genes: {', '.join(genes[:20])}{' ...' if len(genes) > 20 else ''}")
        print(f"  -> {plain.name}  (paste into STRING: https://string-db.org, 'Multiple proteins', Homo sapiens)")
        print(f"  -> {axis}_node_attributes.tsv  (import as node table in Cytoscape)")

    (OUT / "HOWTO_STRING_Cytoscape_cytoHubba.txt").write_text(
        "PPI + hub-gene workflow (per axis)\n"
        "==================================\n\n"
        f"REPRODUCIBILITY RECORD: STRING min score = {STRING_MIN_SCORE:.3f}, "
        f"cytoHubba method = {CYTOHUBBA_METHOD}. Report these in the manuscript.\n\n"
        "1. STRING (https://string-db.org)\n"
        "   - Search > 'Multiple proteins'\n"
        "   - Paste the contents of <axis>_ppi_genes.txt\n"
        "   - Organism: Homo sapiens\n"
        f"   - Minimum required interaction score: {STRING_MIN_SCORE:.3f} (high confidence)\n"
        "   - Export > send network to Cytoscape (stringApp), or download 'network coordinates'\n\n"
        "2. Cytoscape\n"
        "   - Load the STRING network (stringApp) for the axis\n"
        "   - File > Import > Table from file: <axis>_node_attributes.tsv (key column = gene)\n\n"
        "3. cytoHubba (Apps > cytoHubba)\n"
        f"   - Target network: the axis network\n"
        f"   - Compute node scores by: {CYTOHUBBA_METHOD} (recommended), also Degree and MNC\n"
        "   - Export the top 10 (or top 15) ranked nodes = HUB GENES for that axis\n\n"
        "Report hubs per axis. Because the input genes come from enriched pathways,\n"
        "hubs will tend to be the central genes of each axis's dominant program\n"
        "(e.g. interferon hub for severity, chemokine/cytotoxic hub for leakage) -\n"
        "frame them as 'central genes of the axis programme', which is standard.\n")

    print(f"\n[SAVED] -> {OUT}")
    print(f"[NEXT] open {OUT/'HOWTO_STRING_Cytoscape_cytoHubba.txt'} for the manual STRING/Cytoscape steps")
    print("[DONE]")


if __name__ == "__main__":
    main()