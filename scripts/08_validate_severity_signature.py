#!/usr/bin/env python3
"""
============================================================
Dengue Severity Signature Validation with Random Forest

Discovery : GSE215835, DHF vs DF
Signature : Random Forest top-50 genes (from step 04, current run)
External  : GSE43777 GPL570
Validation: direction replication (PRIMARY), cross-validated RF AUC
            (SECONDARY: permutation + bootstrap CI)

Speed tweaks (quality preserved for primary endpoint):
  - N_PERM 1000 -> 250
  - External RF trees 300 -> 100 (secondary endpoint only)
  - Null permutations use 3-fold CV; real AUC keeps up to 5-fold
  - Full AUC + perm + bootstrap only on primary contrast
  - Supplementary contrasts: direction always; AUC once, no perm

Four design fixes retained:
  1. INPUT PATH — current run top-50, not stale importance file
  2. PHASE FILTER — primary = acute only; all-phase supplementary
  3. DIRECTION REPLICATION — primary platform-robust endpoint
  4. severity() token fix — DHF first, word boundaries
============================================================
"""

from pathlib import Path
import gzip
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, binomtest
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import mygene

from paths import (CFG, seed_everything, resolve, read_dir, run_dir,
                   logcpm_path, dataset_tag)

seed_everything()

RUN_ID = None                      # None = current run from _latest.txt
MATRIX = Path("data/raw/GSE43777/GSE43777-GPL570_series_matrix.txt.gz")
RANDOM_STATE = int(CFG["rf"]["random_state"])

# --- speed / quality knobs ---
N_EST_EXTERNAL = 300             
N_PERM = 500                    
N_BOOT = 1000                     
N_SPLITS_AUC = 5                  
N_SPLITS_PERM = 3                  

N_ESTIMATORS = int(
    CFG["rf"].get("n_estimators_by_axis", {}).get(
        "severity", CFG["rf"]["n_estimators"]
    )
)
TOPN = int(CFG["rf"]["top_n"])
PROBE_CACHE = Path("data/processed/GSE43777_GPL570_probe_to_symbol.tsv.gz")

PHASE_SETS = {
    "acute_early+late": ["EarlyAcute", "LateAcute"],
    "acute_early_only": ["EarlyAcute"],
    "all_phases": ["EarlyAcute", "LateAcute", "Convalescent"],
}
PRIMARY = "acute_early+late"


# ============================================================
def parse_series_matrix(path):
    accs, chars, rows, header, table = [], [], [], None, False
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                accs = [x.strip('"') for x in line.rstrip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.rstrip().split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                table = True
            elif line.startswith("!series_matrix_table_end"):
                table = False
            elif table:
                parts = line.rstrip().split("\t")
                if parts[0].strip('"').upper() == "ID_REF":
                    header = [x.strip('"') for x in parts[1:]]
                elif not line.startswith("!"):
                    rows.append(parts)
    n = len(accs)
    meta = pd.DataFrame({"gsm": accs})
    for i, c in enumerate(chars):
        meta[f"char_{i+1}"] = (list(c)[:n] + [""] * (n - len(c)))[:n]
    data = np.array([[np.nan if x in ("", "null") else float(x) for x in r[1:]]
                     for r in rows])
    expr = pd.DataFrame(data, index=[r[0].strip('"') for r in rows], columns=header)
    return meta.set_index("gsm"), expr


def severity(x):
    """FIX 4: check DHF patterns first, use word boundaries."""
    s = str(x).lower()
    if "hemorrhagic" in s or "haemorrhagic" in s or re.search(r"\bdhf\b", s):
        return "DHF"
    if "dengue fever" in s or re.search(r"\bdf\b", s):
        return "DF"
    return "NA"


def phase(x):
    s = str(x).lower()
    if "early acute" in s:
        return "EarlyAcute"
    if "late acute" in s:
        return "LateAcute"
    if "conval" in s:
        return "Convalescent"
    return "NA"


def evaluate_auc(X, y, seed=RANDOM_STATE, n_estimators=N_EST_EXTERNAL, n_splits=None):
    """Leakage-free out-of-fold RF probabilities (secondary endpoint)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    k = n_splits if n_splits is not None else max(2, min(N_SPLITS_AUC, int(np.min(np.bincount(y)))))
    k = max(2, min(k, int(np.min(np.bincount(y)))))
    prob = np.zeros(len(y))
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed + fold,
            n_jobs=-1,
        )
        model.fit(X[tr], y[tr])
        positive_column = list(model.classes_).index(1)
        prob[te] = model.predict_proba(X[te])[:, positive_column]
    return roc_auc_score(y, prob), prob


def probes_to_symbols(expr):
    """Map GPL570 probes to symbols, using a persistent cache and retries."""
    if PROBE_CACHE.exists():
        print(f"  loading cached probe mapping: {PROBE_CACHE}")
        mapping = pd.read_csv(PROBE_CACHE, sep="\t", compression="infer")
    else:
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mapper = mygene.MyGeneInfo()
        probes = expr.index.astype(str).tolist()
        records = []
        batch_size = 500

        for start in range(0, len(probes), batch_size):
            batch = probes[start:start + batch_size]
            result = None
            for attempt in range(5):
                try:
                    result = mapper.querymany(
                        batch,
                        scopes="reporter,affy",
                        fields="symbol",
                        species="human",
                        as_dataframe=False,
                        verbose=False,
                    )
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(
                            f"Probe mapping failed for probes {start + 1}-"
                            f"{start + len(batch)} after 5 attempts"
                        ) from exc
                    wait_seconds = 2 ** attempt
                    print(f"  mapping retry {attempt + 1}/5 after {wait_seconds}s")
                    time.sleep(wait_seconds)

            for hit in result:
                symbol = hit.get("symbol")
                if isinstance(symbol, list):
                    symbol = symbol[0] if symbol else None
                if symbol:
                    records.append({"probe": str(hit.get("query", "")),
                                    "symbol": str(symbol)})
            print(f"  mapped probes {start + 1}-{start + len(batch)}")

        mapping = pd.DataFrame(records).drop_duplicates("probe")
        mapping.to_csv(PROBE_CACHE, sep="\t", index=False, compression="gzip")
        print(f"  probe mapping cached: {PROBE_CACHE}")

    probe_col = "probe" if "probe" in mapping.columns else "query"
    mapping = mapping.dropna(subset=[probe_col, "symbol"]).drop_duplicates(probe_col)
    p2g = dict(zip(mapping[probe_col].astype(str), mapping["symbol"].astype(str)))
    print(f"  probes mapped: {len(p2g)}/{len(expr.index)}")

    mapped = expr.copy()
    mapped["symbol"] = [p2g.get(str(probe), "") for probe in mapped.index]
    mapped = mapped[mapped["symbol"] != ""]
    gene_expr = mapped.groupby("symbol").mean(numeric_only=True)
    print(f"  genes after collapse: {gene_expr.shape[0]}")
    return gene_expr


def discovery_direction(panel):
    """FIX 3: RF gives no direction — take the DHF-vs-DF sign from discovery."""
    tag = dataset_tag("severity")
    e = pd.read_csv(logcpm_path(tag), sep="\t", index_col=0, compression="infer")
    e.index = e.index.astype(str)
    e.columns = e.columns.astype(str)
    dhf = [c for c in e.columns if "_DHF" in c.upper()]
    df_ = [c for c in e.columns if "_DF" in c.upper()]
    g = [x for x in panel if x in e.index]
    fc = e.loc[g, dhf].mean(axis=1) - e.loc[g, df_].mean(axis=1)
    print(f"[discovery] direction from {len(dhf)} DHF vs {len(df_)} DF samples")
    return fc  # positive = up in DHF


def run_contrast(name, phases, meta, gene_expr, panel, disc_fc, out_dir):
    sel = meta[meta["severity"].isin(["DF", "DHF"]) & meta["phase"].isin(phases)]
    sel = sel[sel.index.isin(gene_expr.columns)]
    y = (sel["severity"] == "DHF").astype(int).values
    if y.sum() < 5 or (y == 0).sum() < 5:
        print(f"\n[{name}] skipped — too few per class (DHF={y.sum()}, DF={(y==0).sum()})")
        return None

    present = [g for g in panel if g in gene_expr.index]
    sub = gene_expr.loc[present, sel.index]
    dhf, df_ = sel.index[y == 1], sel.index[y == 0]

    print("\n" + "=" * 70)
    print(f"CONTRAST: {name}   ({'+'.join(phases)})")
    print("=" * 70)
    print(f"DHF={int(y.sum())} DF={int((y==0).sum())} | panel genes present: "
          f"{len(present)}/{len(panel)}")

    # ----- PRIMARY: direction replication (fast) -----
    ext_fc = sub[dhf].mean(axis=1) - sub[df_].mean(axis=1)
    pv = [mannwhitneyu(sub.loc[g, dhf], sub.loc[g, df_],
                       alternative="two-sided").pvalue for g in present]
    discovery_effect = disc_fc.reindex(present).values
    dsign = np.sign(discovery_effect)

    res = pd.DataFrame({
        "gene": present,
        "discovery_logFC_DHF_vs_DF": discovery_effect,
        "disc_sign": dsign,
        "ext_logFC_DHF_vs_DF": ext_fc.values,
        "ext_p": pv,
    })
    res["padj"] = multipletests(np.nan_to_num(pv, nan=1.0), method="fdr_bh")[1]
    res["direction_match"] = np.sign(res["ext_logFC_DHF_vs_DF"]) == res["disc_sign"]

    ok = res["disc_sign"] != 0
    n_match, n_tot = int(res.loc[ok, "direction_match"].sum()), int(ok.sum())
    bt = binomtest(n_match, n_tot, 0.5, alternative="greater") if n_tot else None
    rho, prho = spearmanr(res.loc[ok, "discovery_logFC_DHF_vs_DF"],
                          res.loc[ok, "ext_logFC_DHF_vs_DF"])

    print(f"\n-- PRIMARY: direction replication --")
    print(f"concordance: {n_match}/{n_tot} = {100 * n_match / n_tot:.0f}%  "
          f"binomial p(>chance) = {bt.pvalue:.4f}")
    print(f"Spearman(discovery vs external effect) = {rho:.3f} (p={prho:.3g})")
    print(f"genes significant (p<0.05) AND concordant: "
          f"{int(((res['ext_p'] < 0.05) & res['direction_match']).sum())}")
    print(res.sort_values("ext_p").head(10).to_string(index=False))

    # ----- SECONDARY: RF AUC -----
    is_primary = name == PRIMARY
    auc, prob = evaluate_auc(
        sub.T.values, y,
        n_estimators=N_EST_EXTERNAL,
        n_splits=N_SPLITS_AUC,
    )

    if is_primary:
        # full null + bootstrap only for primary contrast
        rng = np.random.default_rng(RANDOM_STATE)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = evaluate_auc(
                sub.T.values, rng.permutation(y), seed=i,
                n_estimators=N_EST_EXTERNAL,
                n_splits=N_SPLITS_PERM,
            )[0]
        perm_p = (np.sum(null >= auc) + 1) / (N_PERM + 1)
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boot.append(roc_auc_score(y[i], prob[i]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n-- SECONDARY: cross-validated RF panel AUC (full) --")
        print(f"RF trees = {N_EST_EXTERNAL} | AUC = {auc:.3f} | "
              f"permutation p = {perm_p:.4f} "
              f"(null median {np.median(null):.3f}) | 95% CI [{lo:.3f}, {hi:.3f}]")
    else:
        # supplementary: one AUC, skip expensive permutation
        perm_p, lo, hi = np.nan, np.nan, np.nan
        print(f"\n-- SECONDARY: cross-validated RF panel AUC (no perm; supplementary) --")
        print(f"RF trees = {N_EST_EXTERNAL} | AUC = {auc:.3f}")

    res.sort_values("ext_p").to_csv(
        out_dir / f"gene_validation_{name}.tsv", sep="\t", index=False)

    return {
        "contrast": name,
        "phases": "+".join(phases),
        "is_primary": is_primary,
        "DHF": int(y.sum()),
        "DF": int((y == 0).sum()),
        "genes_present": len(present),
        "concordance": n_match / n_tot if n_tot else np.nan,
        "n_match": n_match,
        "n_tested": n_tot,
        "binom_p": bt.pvalue if bt is not None else np.nan,
        "spearman_rho": rho,
        "spearman_p": prho,
        "AUC": auc,
        "permutation_p": perm_p,
        "CI_low": lo,
        "CI_high": hi,
    }


def main():
    print("=" * 70)
    print("STEP 08 — independent-cohort RF signature validation (GSE43777/GPL570)")
    print(f"  speed: N_PERM={N_PERM}, N_EST_EXTERNAL={N_EST_EXTERNAL}, "
          f"perm_folds={N_SPLITS_PERM}")
    print("=" * 70)
    rid = resolve(RUN_ID)
    rf_dir = read_dir(rid, "rf")
    OUT = run_dir(rid, "external_validation", config={
        "external_dataset": "GSE43777_GPL570",
        "top_n": TOPN,
        "primary_contrast": PRIMARY,
        "n_perm": N_PERM,
        "n_boot": N_BOOT,
        "n_estimators_external": N_EST_EXTERNAL,
        "classifier": "RandomForestClassifier",
        "n_estimators_discovery_config": N_ESTIMATORS,
        "model_status": "refit_within_external_cross_validation",
    })

    # FIX 1: current run's top-50
    f = rf_dir / f"severity_top{TOPN}_features.tsv"
    if not f.exists():
        raise SystemExit(f"Missing {f} — run step 04 first.")
    panel = pd.read_csv(f, sep="\t")["gene"].astype(str).tolist()[:TOPN]
    print(f"  panel: {len(panel)} genes from {f}")
    print(f"  first 10: {', '.join(panel[:10])}")

    disc_fc = discovery_direction(panel)

    if not MATRIX.exists():
        raise SystemExit(f"Missing {MATRIX}")
    meta, expr = parse_series_matrix(MATRIX)
    print(f"  external expression: {expr.shape}")

    char_cols = [c for c in meta.columns if c.startswith("char")]
    sev_field = max(char_cols, key=lambda c: (meta[c].map(severity) != "NA").sum())
    phase_field = max(char_cols, key=lambda c: (meta[c].map(phase) != "NA").sum())
    meta["severity"] = meta[sev_field].map(severity)
    meta["phase"] = meta[phase_field].map(phase)
    print(f"  severity <- {sev_field} | phase <- {phase_field}")
    print("\nseverity x phase:")
    print(pd.crosstab(meta["severity"], meta["phase"]))

    gene_expr = probes_to_symbols(expr)
    missing = [g for g in panel if g not in gene_expr.index]
    print(f"  panel present on GPL570: {len(panel) - len(missing)}/{len(panel)}"
          + (f" | missing: {missing}" if missing else ""))

    rows = []
    for name in [PRIMARY] + [k for k in PHASE_SETS if k != PRIMARY]:
        r = run_contrast(name, PHASE_SETS[name], meta, gene_expr, panel, disc_fc, OUT)
        if r:
            rows.append(r)

    summ = pd.DataFrame(rows)
    summ.insert(0, "dataset", "GSE43777_GPL570")
    summ.insert(1, "signature", f"RF_top{TOPN}_severity")
    summ.to_csv(OUT / "external_validation_summary.tsv", sep="\t", index=False)

    print("\n" + "=" * 70 + "\nSUMMARY (report the primary row)\n" + "=" * 70)
    print(summ[["contrast", "is_primary", "DHF", "DF", "genes_present",
                "concordance", "binom_p", "AUC", "permutation_p"]].round(4).to_string(index=False))
    print("\nDirection replication is the primary evidence: it is robust to the")
    print("RNA-seq -> microarray platform change. Cross-validated RF AUC is secondary.")
    print("The genes were fixed before validation; the RF was fitted only inside")
    print("GSE43777 CV folds. Call this independent-cohort signature validation,")
    print("not validation of a locked discovery-trained model.")
    print(f"\n[SAVED] -> {OUT}\n[DONE]")


if __name__ == "__main__":
    main()
