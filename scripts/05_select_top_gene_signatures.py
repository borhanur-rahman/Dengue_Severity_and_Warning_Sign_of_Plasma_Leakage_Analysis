#!/usr/bin/env python3
"""Step 04 — RF top-50 signature per axis + cross-axis overlap.

Produces the top-50 gene lists that steps 05/06 consume, written to
v2/runs/<RUN_ID>/rf/{axis}_top50_features.tsv  (and {axis}_top50_genes.tsv).

METHOD == your original top50_overlap script, UNCHANGED:
  1. top-2000 HVGs (variance rank + MAD rank) on each axis's own samples
  2. RF rank = Gini (n_seed_fits) + permutation importance, ranks summed
  3. take TOP-50, REFIT RF on those 50, CV (LOOCV if n<20 else RepStrat5x10)
  4. leakage restricted to NovaSeq6000 (discovery)
  5. cross-axis common genes with a hypergeometric p

REVIEWER-BLOCKING FIXES (do NOT change the discovery gene lists; written as
SEPARATE supplementary files so reviewers get an honest estimate):
  * external validation: train top-50 on NovaSeq, test once on held-out HiSeq
    -> {leakage}_external_validation.tsv   [REVIEWER_DEFENCE 1.1]
  * nested CV: HVG + ranking + top-50 chosen INSIDE each fold
    -> {axis}_nested_cv.tsv                 [REVIEWER_DEFENCE 1.1]
  * overlap universe = INTERSECTION of the two datasets  [REVIEWER_DEFENCE 2.1]
  * global seeding for identical gene lists
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             accuracy_score, matthews_corrcoef)

from paths import (CFG, seed_everything, make_run_id, run_dir, set_latest,
                      logcpm_path, metadata_path, dataset_tag)

seed_everything()

N_HVG = int(CFG["rf"]["hvg_n"])
TOPN = int(CFG["rf"]["top_n"])          # PRIMARY K (steps 05/06 read this)
K_GRID = sorted(set(int(k) for k in CFG["rf"].get("k_grid", [TOPN])) | {TOPN})
N_ESTIMATORS = int(CFG["rf"]["n_estimators"])          # fallback
N_EST_BY_AXIS = CFG["rf"].get("n_estimators_by_axis", {}) or {}
ACTIVE_N_EST = N_ESTIMATORS                            # set per axis in axis_signature
RANK_SEED_FITS = int(CFG["rf"]["n_seed_fits"])
RS = int(CFG["rf"]["random_state"])
DO_EXTVAL = bool(CFG["rf"].get("external_validation", True))
DO_NESTED = bool(CFG["rf"].get("nested_cv", True))
UNIVERSE = CFG["enrichment"].get("overlap_universe", "intersection")
AXES = list(CFG["axes"].keys())
TARGET_GENES = CFG["target_genes"]

CASE_TOKENS = ["d+w", "with warning", "warning signs"]
CTRL_TOKENS = ["d-w", "without warning", "no warning"]

RUN_ID = make_run_id()
OUT: Path = None


def load_expr(tag):
    e = pd.read_csv(logcpm_path(tag), sep="\t", index_col=0, compression="infer")
    e.index = e.index.astype(str); e.columns = e.columns.astype(str)
    return e


def top_hvg(expr, samples, n):
    sub = expr[[s for s in samples if s in expr.columns]]
    var = sub.var(axis=1)
    mad = sub.sub(sub.median(axis=1), axis=0).abs().median(axis=1)
    return (var.rank(ascending=False) + mad.rank(ascending=False)).sort_values().head(n).index.tolist()


def plat_key(v):
    s = str(v).lower()
    if "hiseq" in s: return "HiSeq2500"
    if "nova" in s: return "NovaSeq6000"
    return str(v)


def get_samples(axis):
    """Return discovery (samples,y) and held-out (samples,y) for the axis."""
    tag = dataset_tag(axis)
    expr = load_expr(tag)
    spec = CFG["axes"][axis]
    if axis == "severity":
        s, y = [], []
        for c in expr.columns:
            cu = c.upper()
            if "_DHF" in cu: s.append(c); y.append(1)
            elif "_DF" in cu: s.append(c); y.append(0)
        return expr, s, np.array(y), [], np.array([])
    # leakage
    meta = pd.read_csv(metadata_path(tag), sep="\t"); meta["sample_id"] = meta["sample_id"].astype(str)
    m = meta[meta["axis"].astype(str).str.lower() == "leakage"] if "axis" in meta.columns else meta
    if "cell_subtype" in m.columns:
        pb = m["cell_subtype"].astype(str).str.upper().str.contains("PBMC")
        if pb.any(): m = m[pb]
    m = m[m["sample_id"].isin(expr.columns)]
    def code(v):
        s = str(v).lower()
        if any(t in s for t in CASE_TOKENS): return 1
        if any(t in s for t in CTRL_TOKENS): return 0
        return -1
    m = m.assign(y=m["condition"].map(code)); m = m[m["y"] >= 0]
    m["plat"] = m["platform"].map(plat_key)
    disc = spec["discovery_platform"]; val = spec["validation_platform"]
    dm = m[m["plat"] == disc]; vm = m[m["plat"] == val]
    return (expr, dm["sample_id"].tolist(), dm["y"].to_numpy(),
            vm["sample_id"].tolist(), vm["y"].to_numpy())


def make_rf(seed):
    """Uses ACTIVE_N_EST, which axis_signature sets to the per-axis value from
    rf.n_estimators_by_axis (chosen by ranking convergence in step 03b)."""
    return RandomForestClassifier(n_estimators=ACTIVE_N_EST, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def rank_by_rf(X, y, seeds=RANK_SEED_FITS, perm=True):
    gini = np.zeros(X.shape[1])
    for s in range(seeds):
        gini += make_rf(RS + s).fit(X, y).feature_importances_
    gini /= seeds
    if perm:
        pr = permutation_importance(make_rf(RS).fit(X, y), X, y, n_repeats=10,
                                    random_state=RS, n_jobs=-1, scoring="balanced_accuracy")
        order = (pd.Series(gini).rank(ascending=False) +
                 pd.Series(pr.importances_mean).rank(ascending=False))
        return order.sort_values().index.to_numpy(), gini, pr.importances_mean
    order = pd.Series(gini).rank(ascending=False).sort_values().index.to_numpy()
    return order, gini, np.full(X.shape[1], np.nan)


def cv_eval(X, y):
    n = len(y)
    if n < 20:
        splits = LeaveOneOut().split(X); scheme = "LOOCV"
    else:
        splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS).split(X, y)
        scheme = "RepeatedStratified5x10"
    prob = np.zeros(n); cnt = np.zeros(n)
    for tr, te in splits:
        prob[te] += make_rf(RS).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; cnt[te] += 1
    oof = prob / np.maximum(cnt, 1); pred = (oof >= 0.5).astype(int)
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else np.nan
    return dict(scheme=scheme, n=n, accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred), auc=auc,
                mcc=matthews_corrcoef(y, pred))


def external_validation(axis, expr, disc_s, disc_y, val_s, val_y, ranked):
    if len(val_s) < 3 or len(np.unique(val_y)) < 2:
        return None
    genes = [g for g in ranked[:TOPN] if g in expr.index]
    Xtr = expr.loc[genes, disc_s].T.values
    Xte = expr.loc[genes, val_s].T.values
    rf = make_rf(RS).fit(Xtr, disc_y)
    prob = rf.predict_proba(Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
    res = dict(axis=axis, heldout_platform=CFG["axes"][axis]["validation_platform"],
               heldout_n=len(val_y), n_genes=len(genes),
               accuracy=accuracy_score(val_y, pred),
               balanced_accuracy=balanced_accuracy_score(val_y, pred),
               auc=roc_auc_score(val_y, prob), mcc=matthews_corrcoef(val_y, pred),
               majority=float(max(val_y.mean(), 1 - val_y.mean())))
    print(f"[{axis}] EXTERNAL {res['heldout_platform']} (n={len(val_y)}, untouched): "
          f"bal_acc={res['balanced_accuracy']:.3f} auc={res['auc']:.3f} "
          f"(majority {res['majority']:.3f})")
    print("       ^ nothing was fitted to this platform — strongest number here.")
    return res


def nested_cv(axis, expr, samples, y):
    if len(np.unique(y)) < 2 or len(y) < 8:
        print(f"[{axis}] nested CV skipped (n={len(y)} too small)")
        return None
    counts = np.bincount(y); ns = max(2, min(CFG["rf"]["nested_outer_splits"], int(counts[counts>0].min())))
    outer = RepeatedStratifiedKFold(n_splits=ns, n_repeats=CFG["rf"]["nested_outer_repeats"], random_state=RS)
    idx = np.arange(len(samples))
    prob = np.zeros(len(samples)); cnt = np.zeros(len(samples))
    for tr, te in outer.split(idx, y):
        tr_s = [samples[i] for i in tr]; te_s = [samples[i] for i in te]
        genes = top_hvg(expr, tr_s, N_HVG)                      # train only
        Xtr = expr.loc[genes, tr_s].T.values; Xte = expr.loc[genes, te_s].T.values
        order, _, _ = rank_by_rf(Xtr, y[tr], seeds=5, perm=False)   # Gini-only in-fold for speed
        g = order[:TOPN]
        rf = make_rf(RS).fit(Xtr[:, g], y[tr])
        prob[te] += rf.predict_proba(Xte[:, g])[:, 1]; cnt[te] += 1
    oof = prob / np.maximum(cnt, 1); pred = (oof >= 0.5).astype(int)
    res = dict(axis=axis, scheme=f"nested_{ns}x{CFG['rf']['nested_outer_repeats']}",
               balanced_accuracy=balanced_accuracy_score(y, pred),
               auc=roc_auc_score(y, oof) if len(np.unique(y))>1 else np.nan,
               mcc=matthews_corrcoef(y, pred))
    print(f"[{axis}] NESTED CV (selection inside folds): "
          f"bal_acc={res['balanced_accuracy']:.3f} auc={res['auc']:.3f}")
    return res


def axis_signature(axis, expr, disc_s, disc_y, val_s, val_y):
    global ACTIVE_N_EST
    ACTIVE_N_EST = int(N_EST_BY_AXIS.get(axis, N_ESTIMATORS))
    print("\n" + "=" * 70 + f"\nAXIS: {axis}"
          + (f"  (discovery: {CFG['axes'][axis]['discovery_platform']})" if val_s else "")
          + "\n" + "=" * 70)
    genes = top_hvg(expr, disc_s, N_HVG)
    X = expr.loc[genes, disc_s].T.values
    print(f"[{axis}] {len(genes)} HVGs x {len(disc_s)} samples "
          f"(case {int(disc_y.sum())}/ctrl {int((disc_y==0).sum())}) "
          f"| n_estimators = {ACTIVE_N_EST}"
          + ("" if axis in N_EST_BY_AXIS else "  [fallback — run 3b_tune_rf.py]"))
    order, gini, perm = rank_by_rf(X, disc_y)
    ranked = [genes[i] for i in order]
    top50 = ranked[:TOPN]

    # the files steps 05/06 read
    pd.DataFrame({"gene": top50, "rank": range(1, len(top50)+1),
                  "gini": [gini[i] for i in order[:TOPN]],
                  "perm": [perm[i] for i in order[:TOPN]]}).to_csv(
        OUT / f"{axis}_top{TOPN}_features.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": top50, "rank": range(1, len(top50)+1)}).to_csv(
        OUT / f"{axis}_top{TOPN}_genes.tsv", sep="\t", index=False)

    perf = cv_eval(X[:, order[:TOPN]], disc_y)
    print(f"[{axis}] REFIT top-{TOPN}: {perf['scheme']} bal_acc={perf['balanced_accuracy']:.3f} "
          f"auc={perf['auc']:.3f} mcc={perf['mcc']:.3f}  (PRIMARY, in-sample)")

    # --- supplementary K sweep: refit RF on each top-K and cross-validate ---
    print(f"[{axis}] K sweep {K_GRID} (supplementary; primary remains K={TOPN}):")
    sweep_rows = []
    for K in [k for k in K_GRID if k <= len(ranked)]:
        sc = cv_eval(X[:, order[:K]], disc_y)
        sweep_rows.append({"axis": axis, "top_k": K, "is_primary": K == TOPN, **sc})
        print(f"    top-{K:<4d} bal_acc={sc['balanced_accuracy']:.3f} "
              f"auc={sc['auc']:.3f} acc={sc['accuracy']:.3f} mcc={sc['mcc']:.3f}")
        pd.DataFrame({"gene": ranked[:K], "rank": range(1, K + 1)}).to_csv(
            OUT / f"{axis}_top{K}_genes_grid.tsv", sep="\t", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / f"{axis}_k_sweep.tsv", sep="\t", index=False)

    # --- supplementary honest estimates (do not affect top50) ---
    if DO_NESTED:
        nr = nested_cv(axis, expr, disc_s, disc_y)
        if nr: pd.DataFrame([nr]).to_csv(OUT / f"{axis}_nested_cv.tsv", sep="\t", index=False)
    if DO_EXTVAL and val_s:
        er = external_validation(axis, expr, disc_s, disc_y, val_s, val_y, ranked)
        if er: pd.DataFrame([er]).to_csv(OUT / f"{axis}_external_validation.tsv", sep="\t", index=False)

    hits = [(g, ranked.index(g)+1) for g in TARGET_GENES if g in ranked]
    if hits:
        print(f"[{axis}] target-gene ranks: " + ", ".join(f"{g}={r}" for g,r in sorted(hits,key=lambda x:x[1])))
    return set(top50), perf, ranked, sweep


def make_figure(sweeps, grid, a1, a2):
    """Top-K vs accuracy for BOTH axes in one figure (+ overlap panel)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip figure] matplotlib unavailable: {exc}")
        return

    colours = {a1: "#2b6cb0", a2: "#c05621"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    for axis, sw in sweeps.items():
        sw = sw.sort_values("top_k")
        ax[0].plot(sw["top_k"], sw["balanced_accuracy"], marker="o", lw=2,
                   color=colours.get(axis), label=f"{axis} — balanced accuracy")
        ax[0].plot(sw["top_k"], sw["auc"], marker="s", ls="--", lw=1.6, alpha=0.75,
                   color=colours.get(axis), label=f"{axis} — AUC")
    ax[0].axvline(TOPN, color="grey", ls=":", lw=1.5)
    ax[0].annotate(f"primary K={TOPN}", xy=(TOPN, ax[0].get_ylim()[0]),
                   xytext=(4, 6), textcoords="offset points", fontsize=8, color="grey")
    ax[0].axhline(0.5, color="black", ls=":", lw=1, alpha=0.5)
    ax[0].set_xlabel("Top-K RF genes")
    ax[0].set_ylabel("Cross-validated performance")
    ax[0].set_title("Top-K vs performance (both axes)")
    ax[0].set_xticks(sorted(sweeps[a1]["top_k"]))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    ax[1].bar([str(k) for k in grid["top_k"]], grid["n_common"], color="#2b6cb0",
              label="observed common genes")
    ax[1].plot([str(k) for k in grid["top_k"]], grid["expected_by_chance"],
               color="crimson", marker="o", lw=1.8, label="expected by chance")
    ax[1].set_xlabel("Top-K")
    ax[1].set_ylabel("Common genes")
    ax[1].set_title(f"{a1} top-K  vs  {a2} top-K overlap")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"topk_vs_accuracy.{ext}", dpi=300)
    plt.close(fig)
    print(f"\n  [figure] {OUT}/topk_vs_accuracy.png (+ .pdf, 300 dpi)")


def main():
    global OUT
    OUT = run_dir(RUN_ID, "rf", config=CFG["rf"] | {
        "axes": AXES, "n_estimators_resolved": {
            a: int(N_EST_BY_AXIS.get(a, N_ESTIMATORS)) for a in AXES}})
    set_latest(RUN_ID)
    print("=" * 70 + f"\nSTEP 04 - RF top-{TOPN} signature   run={RUN_ID}\n" + "=" * 70)
    print(f"  output: {OUT}")

    sets, perfs, univ, rankings, sweeps = {}, {}, {}, {}, {}
    for axis in AXES:
        expr, ds, dy, vs, vy = get_samples(axis)
        univ[axis] = set(expr.index)
        if len(ds) < 6 or len(np.unique(dy)) < 2:
            raise SystemExit(f"[ERROR] {axis}: too few samples")
        sets[axis], perfs[axis], rankings[axis], sweeps[axis] = axis_signature(
            axis, expr, ds, dy, vs, vy)

    a1, a2 = AXES
    M_univ = (len(univ[a1] & univ[a2]) if UNIVERSE == "intersection"
              else len(univ[a1] | univ[a2]))

    # --- eligibility: how much of each top-K can overlap at all? ---
    # HVGs are selected in each axis's OWN universe, so part of each top-K may
    # be absent from the other dataset and can never overlap. The hypergeometric
    # null assumes both draws come from the shared universe, so reporting this
    # makes explicit that the null is CONSERVATIVE.
    print("\n" + "=" * 70 + "\nOVERLAP ELIGIBILITY\n" + "=" * 70)
    print(f"  {a1} universe {len(univ[a1])} | {a2} universe {len(univ[a2])} | "
          f"shared {M_univ}")
    elig_rows = []
    for K in [k for k in K_GRID if all(k <= len(rankings[a]) for a in AXES)]:
        e1 = sum(1 for g in rankings[a1][:K] if g in univ[a2])
        e2 = sum(1 for g in rankings[a2][:K] if g in univ[a1])
        elig_rows.append({"top_k": K, f"{a1}_eligible": e1, f"{a2}_eligible": e2,
                          f"{a1}_eligible_pct": round(100 * e1 / K, 1),
                          f"{a2}_eligible_pct": round(100 * e2 / K, 1),
                          "expected_hypergeom": round(K * K / M_univ, 3),
                          "expected_eligibility_adjusted": round(e1 * e2 / M_univ, 3)})
        print(f"  top-{K:<4d} {a1}: {e1}/{K} ({100*e1/K:.0f}%) measurable in {a2}  |  "
              f"{a2}: {e2}/{K} ({100*e2/K:.0f}%) measurable in {a1}")
    elig = pd.DataFrame(elig_rows)
    elig.to_csv(OUT / "overlap_eligibility.tsv", sep="\t", index=False)
    print("  -> hypergeometric expectation assumes full eligibility, so the")
    print("     reported p-values are conservative (see overlap_eligibility.tsv)")

    # ---------- supplementary: common genes at EVERY K ----------
    print("\n" + "=" * 70 + "\nCROSS-AXIS COMMON GENES BY K (supplementary)\n" + "=" * 70)
    grid_rows, grid_genes = [], []
    for K in [k for k in K_GRID if all(k <= len(rankings[a]) for a in AXES)]:
        s1, s2 = set(rankings[a1][:K]), set(rankings[a2][:K])
        com = sorted(s1 & s2, key=lambda g: (rankings[a1].index(g) + rankings[a2].index(g), g))
        exp_c = K * K / M_univ if M_univ else float("nan")
        p_k = hypergeom.sf(len(com) - 1, M_univ, K, K) if com and M_univ else 1.0
        b1 = sweeps[a1].set_index("top_k").loc[K] if K in set(sweeps[a1]["top_k"]) else None
        b2 = sweeps[a2].set_index("top_k").loc[K] if K in set(sweeps[a2]["top_k"]) else None
        grid_rows.append({
            "top_k": K, "n_common": len(com),
            "overlap_percent": 100.0 * len(com) / K,
            "jaccard": len(com) / len(s1 | s2) if (s1 | s2) else float("nan"),
            "expected_by_chance": round(exp_c, 3),
            "fold_enrichment": (len(com) / exp_c) if exp_c else float("nan"),
            "hypergeometric_p": p_k,
            f"{a1}_bal_acc": (float(b1["balanced_accuracy"]) if b1 is not None else float("nan")),
            f"{a1}_auc": (float(b1["auc"]) if b1 is not None else float("nan")),
            f"{a2}_bal_acc": (float(b2["balanced_accuracy"]) if b2 is not None else float("nan")),
            f"{a2}_auc": (float(b2["auc"]) if b2 is not None else float("nan")),
            "common_genes": ",".join(com)})
        for i, g in enumerate(com, 1):
            grid_genes.append({"top_k": K, "common_rank": i, "gene": g,
                               f"{a1}_rank": rankings[a1].index(g) + 1,
                               f"{a2}_rank": rankings[a2].index(g) + 1})
        pd.DataFrame([r for r in grid_genes if r["top_k"] == K]).to_csv(
            OUT / f"common_top{K}_genes.tsv", sep="\t", index=False)
        print(f"  top-{K:<4d} common={len(com):<3d} expected={exp_c:5.2f} "
              f"fold={(len(com)/exp_c if exp_c else float('nan')):5.2f}x p={p_k:.3g}"
              + (f"  [{', '.join(com[:8])}{' ...' if len(com) > 8 else ''}]" if com else ""))

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "common_genes_by_k.tsv", sep="\t", index=False)
    pd.DataFrame(grid_genes).to_csv(OUT / "common_genes_by_k_long.tsv", sep="\t", index=False)
    pd.concat(sweeps.values(), ignore_index=True).to_csv(
        OUT / "k_sweep_all_axes.tsv", sep="\t", index=False)
    make_figure(sweeps, grid, a1, a2)

    common = sorted(sets[a1] & sets[a2]); union = sorted(sets[a1] | sets[a2])
    M = len(univ[a1] & univ[a2]) if UNIVERSE == "intersection" else len(univ[a1] | univ[a2])
    expected = TOPN * TOPN / M if M else float("nan")
    p_over = hypergeom.sf(len(common) - 1, M, TOPN, TOPN) if common and M else 1.0
    fold = len(common) / expected if expected else float("nan")

    print("\n" + "=" * 70 + f"\nCROSS-AXIS COMMON TOP-{TOPN}\n" + "=" * 70)
    print(f"universe ({UNIVERSE}): {M} | expected: {expected:.2f}")
    print(f"common: {len(common)} | fold: {fold:.2f}x | hypergeometric p: {p_over:.3g}")
    print(f"common genes: {', '.join(common) if common else '(none)'}")

    pd.DataFrame({"gene": common}).to_csv(OUT / f"common_top{TOPN}.tsv", sep="\t", index=False)
    pd.DataFrame([{"top_n": TOPN, "universe": UNIVERSE, "M": M,
                   "common": len(common), "expected": round(expected,2),
                   "fold_enrichment": round(fold,2), "hypergeometric_p": p_over,
                   f"{a1}_bal_acc": perfs[a1]["balanced_accuracy"],
                   f"{a2}_bal_acc": perfs[a2]["balanced_accuracy"],
                   "common_genes": ",".join(common) if common else "-"}]).to_csv(
        OUT / "overlap_summary.tsv", sep="\t", index=False)
    print(f"\n[SAVED] -> {OUT}\n[DONE]  next: 5_enrichment.py")


if __name__ == "__main__":
    main()