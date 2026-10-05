#!/usr/bin/env python3
"""Step 02 — Ensembl->symbol mapping, drop pseudogenes, symbol cleanup.

  * mygene mapping is cached to disk with the query date, so re-runs stay
    reproducible;
  * unmapped Ensembl IDs are saved to a separate file instead of dropped;
  * symbol-cleanup regexes spare the S6 kinases RPS6KA1/KB1 and genes such as
    MTOR/MTHFR/HBEGF/HBP1 — MT- is matched exactly, globins explicitly.
The mapping choice, the pseudogene rule, and the removal categories are
unchanged.
"""
from pathlib import Path
import datetime as dt
import re

import numpy as np
import pandas as pd
import mygene

from paths import (CFG, seed_everything, ensure_data_dirs, PROC_DIR, QC_DIR,
                      metadata_path, counts_path)

seed_everything()
ensure_data_dirs()

prep_dir = QC_DIR

gse178240_counts = Path(CFG["raw_data"]["gse178240_counts"])
gse215835_counts = Path(CFG["raw_data"]["gse215835_counts"])
meta178_path = metadata_path("GSE178240")
meta215_path = metadata_path("GSE215835")

DROP_PSEUDO = bool(CFG["preprocessing"]["drop_pseudogenes"])
MAP_CACHE = PROC_DIR / "ensembl_symbol_map.tsv.gz"

mg = mygene.MyGeneInfo()

# --- symbol cleanup: MT-, RP[LS]\d, MRP[LS]\d patterns and globins; RPS6K kinases spared ---
_sc = CFG["preprocessing"]["symbol_cleanup"]
MITO_RE = re.compile(_sc["mito_regex"])
RIBO_RE = re.compile(_sc["ribosomal_regex"])
RIBO_SPARE_RE = re.compile(_sc["ribosomal_spare_regex"])
MITORIBO_RE = re.compile(_sc["mito_ribosomal_regex"])
GLOBIN = {g.upper() for g in _sc["globin"]}
EXACT = {g.upper() for g in CFG["preprocessing"]["unwanted_gene_exact"]}


def is_unwanted(g):
    s = str(g).upper()
    if s in EXACT or s in GLOBIN:
        return True
    if MITO_RE.match(s) or MITORIBO_RE.match(s):
        return True
    if RIBO_RE.match(s) and not RIBO_SPARE_RE.match(s):
        return True
    return False


def to_numeric_clean(expr):
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    return expr.loc[expr.sum(axis=1) > 0]


def collapse_symbols(expr):
    expr = expr.copy(); expr.index = expr.index.astype(str)
    expr = expr[~expr.index.isin(["", "nan", "None", "NaN"])]
    n_dup = int(expr.index.duplicated().sum())
    expr = expr.groupby(expr.index).sum()
    print(f"    duplicate symbols collapsed: {n_dup} -> {expr.shape[0]} unique genes")
    return expr


def remove_unwanted(expr, tag):
    mask = expr.index.to_series().apply(is_unwanted)
    pd.DataFrame({"gene_symbol": expr.index[mask].tolist()}).to_csv(
        prep_dir / f"{tag}_removed_symbol_cleanup.tsv", sep="\t", index=False)
    print(f"    symbol cleanup (mito/ribo/globin) removed: {int(mask.sum())}")
    return expr.loc[~mask]


def map_ensembl_cached(ens):
    ens = sorted(set(ens))
    if MAP_CACHE.exists():
        cached = pd.read_csv(MAP_CACHE, sep="\t")
        have = set(cached["query"].astype(str))
        todo = [e for e in ens if e not in have]
        print(f"    mygene cache: {len(ens)-len(todo)}/{len(ens)} hit")
    else:
        cached, todo = pd.DataFrame(), ens
    if todo:
        res = mg.querymany(todo, scopes="ensembl.gene", fields="symbol,type_of_gene",
                           species="human", as_dataframe=True, df_index=False, verbose=False)
        res["query"] = res["query"].astype(str).str.split(".").str[0]
        for c in ("symbol", "type_of_gene"):
            if c not in res.columns: res[c] = np.nan
        res = res[["query", "symbol", "type_of_gene"]]
        res["queried_on"] = dt.date.today().isoformat()
        cached = pd.concat([cached, res], ignore_index=True) if len(cached) else res
        cached.to_csv(MAP_CACHE, sep="\t", index=False, compression="gzip")
    return cached[cached["query"].isin(ens)].copy()


def biotype_pseudo_symbols(symbols):
    res = mg.querymany(list(symbols), scopes="symbol", fields="type_of_gene",
                       species="human", as_dataframe=True, df_index=False, verbose=False)
    if "type_of_gene" not in res.columns:
        return set()
    res["is_pseudo"] = res["type_of_gene"].astype(str).eq("pseudo")
    grp = res.groupby("query")["is_pseudo"].all()
    return set(grp[grp].index.astype(str))


def prepare_gse178240():
    print("\n" + "=" * 70 + "\n[GSE178240] Ensembl -> symbol, drop pseudogenes\n" + "=" * 70)
    meta = pd.read_csv(meta178_path, sep="\t")
    samples = meta["sample_id"].astype(str).tolist()
    raw = pd.read_csv(gse178240_counts, sep="\t", low_memory=False)
    gene_col = raw.columns[0]
    missing = [s for s in samples if s not in raw.columns]
    if missing:
        raise ValueError(f"samples not in counts: {missing[:5]}")
    expr = to_numeric_clean(raw[[gene_col] + samples].set_index(gene_col))
    print(f"[GSE178240] raw genes x samples: {expr.shape}")

    ens_full = [str(x) for x in expr.index]
    ens = [e.split(".")[0] for e in ens_full]
    m = map_ensembl_cached(ens)
    m = m[m["symbol"].notna()].copy()
    m["is_pseudo"] = m["type_of_gene"].astype(str).eq("pseudo")
    m = m.sort_values("is_pseudo").drop_duplicates("query", keep="first")
    sym = dict(zip(m["query"], m["symbol"]))
    pseudo = set(m.loc[m["is_pseudo"], "query"])
    print(f"[GSE178240] mapped: {len(sym)} | pseudogenes flagged: {len(pseudo)}")

    unmapped = sorted({e for e in ens if e not in sym})
    pd.DataFrame({"ensembl": unmapped}).to_csv(
        prep_dir / "GSE178240_unmapped_ensembl.tsv", sep="\t", index=False)
    print(f"[GSE178240] unmapped IDs saved: {len(unmapped)}")

    keep_rows, keep_syms = [], []
    for full_id, e in zip(ens_full, ens):
        if DROP_PSEUDO and e in pseudo:
            continue
        s = sym.get(e)
        if s is not None:
            keep_rows.append(full_id); keep_syms.append(s)
    expr = expr.loc[keep_rows]; expr.index = keep_syms
    print(f"[GSE178240] after mapping + pseudo drop: {expr.shape}")

    expr = remove_unwanted(collapse_symbols(expr), "GSE178240")
    expr.index.name = "gene_symbol"
    out = str(counts_path("GSE178240"))
    expr.to_csv(out, sep="\t", compression="gzip")
    print(f"[GSE178240] FINAL: {expr.shape} -> {out}")
    return expr


def prepare_gse215835():
    print("\n" + "=" * 70 + "\n[GSE215835] symbols; drop pseudogenes\n" + "=" * 70)
    meta = pd.read_csv(meta215_path, sep="\t")
    samples = meta["sample_id"].astype(str).tolist()
    raw = pd.read_csv(gse215835_counts, sep="\t", low_memory=False)
    gene_col = "Symbol" if "Symbol" in raw.columns else raw.columns[0]
    missing = [s for s in samples if s not in raw.columns]
    if missing:
        raise ValueError(f"samples not in counts: {missing}")
    expr = raw[[gene_col] + samples].copy()
    expr[gene_col] = expr[gene_col].astype(str)
    expr = to_numeric_clean(expr.set_index(gene_col))
    print(f"[GSE215835] raw genes x samples: {expr.shape}")

    if DROP_PSEUDO:
        pseudo = biotype_pseudo_symbols(set(expr.index))
        before = expr.shape[0]
        expr = expr.loc[~expr.index.isin(pseudo)]
        print(f"[GSE215835] pseudogenes dropped: {before - expr.shape[0]}")

    expr = remove_unwanted(collapse_symbols(expr), "GSE215835")
    expr.index.name = "gene_symbol"
    out = str(counts_path("GSE215835"))
    expr.to_csv(out, sep="\t", compression="gzip")
    print(f"[GSE215835] FINAL: {expr.shape} -> {out}")
    return expr


def main():
    e178 = prepare_gse178240()
    e215 = prepare_gse215835()
    common = sorted(set(e178.index) & set(e215.index))
    pd.DataFrame({"gene_symbol": common}).to_csv(
        prep_dir / "genes_common_to_both_datasets.tsv", sep="\t", index=False)
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(f"GSE178240 genes: {e178.shape[0]} | GSE215835 genes: {e215.shape[0]}")
    print(f"common: {len(common)}")
    print("[DONE] Step 02 completed.")


if __name__ == "__main__":
    main()