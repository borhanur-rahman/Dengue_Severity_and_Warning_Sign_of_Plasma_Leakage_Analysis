#!/usr/bin/env python3
"""
Step 03b — select n_estimators per axis by OUT-OF-BAG ERROR.

READ ONLY. Prints a config block to paste; changes nothing on its own.

Selection rule
--------------
For each axis, on the top-2000 HVGs of its discovery samples, fit a Random
Forest at each n_estimators in GRID and record the out-of-bag error. The
recommended value is the n_estimators with the LOWEST mean OOB error.

Two things are done to make that rule trustworthy:

1. SEED AVERAGING. A single forest's OOB error is noisy, so each grid point is
   fitted with N_SEEDS forests and the mean +/- sd is reported. Selecting on a
   single seed would pick the luckiest seed rather than the best setting.

2. GRANULARITY WARNING. OOB error can only take values k/n, so its resolution
   is 1/n. At n = 11 that is 9.1% -- one sample changing side moves the curve
   by more than any real difference between tree counts. The script prints the
   resolution per axis and flags when the spread across the whole grid is
   smaller than one step, i.e. when the ranking of grid values is not
   meaningful.

Output: a tuning table (rf_oob_tuning.tsv) and a 300 dpi figure
(oob_error_vs_n_estimators.png/.pdf) showing OOB error against forest size for
each axis, with the seed spread, the selected value, and a shaded band one OOB
step wide above the minimum -- values inside that band cannot be distinguished
given the resolution of the metric.

Also reported: the smallest n_estimators whose mean OOB error is within one
standard error of the minimum. Random Forest OOB error decreases toward an
asymptote rather than rising again (Breiman 2001), so the minimum is often on a
plateau; the parsimonious value is given for reference but the LOWEST-error
value is what is recommended, as requested.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from paths import (CFG, seed_everything, logcpm_path, metadata_path,
                      dataset_tag, QC_DIR)

seed_everything()

N_HVG = int(CFG["rf"]["hvg_n"])
RS = int(CFG["rf"]["random_state"])
AXES = list(CFG["axes"].keys())

GRID = [100, 200, 300, 500, 750, 1000, 1500, 2000]
N_SEEDS = 10                  # forests averaged per grid point

CASE_TOKENS = ["d+w", "with warning", "warning signs"]
CTRL_TOKENS = ["d-w", "without warning", "no warning"]


def load_expr(tag):
    e = pd.read_csv(logcpm_path(tag), sep="\t", index_col=0, compression="infer")
    e.index = e.index.astype(str); e.columns = e.columns.astype(str)
    return e


def top_hvg(expr, samples, n):
    sub = expr[[s for s in samples if s in expr.columns]]
    var = sub.var(axis=1)
    mad = sub.sub(sub.median(axis=1), axis=0).abs().median(axis=1)
    return (var.rank(ascending=False) + mad.rank(ascending=False)
            ).sort_values().head(n).index.tolist()


def plat_key(v):
    s = str(v).lower()
    if "hiseq" in s: return "HiSeq2500"
    if "nova" in s: return "NovaSeq6000"
    return str(v)


def discovery_samples(axis):
    tag = dataset_tag(axis)
    expr = load_expr(tag)
    if axis == "severity":
        s, y = [], []
        for c in expr.columns:
            cu = c.upper()
            if "_DHF" in cu: s.append(c); y.append(1)
            elif "_DF" in cu: s.append(c); y.append(0)
        return expr, s, np.array(y)
    meta = pd.read_csv(metadata_path(tag), sep="\t")
    meta["sample_id"] = meta["sample_id"].astype(str)
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
    disc = CFG["axes"][axis]["discovery_platform"]
    dm = m[m["plat"] == disc] if disc else m
    return expr, dm["sample_id"].tolist(), dm["y"].to_numpy()


def oob_error(X, y, n_est, seed):
    m = RandomForestClassifier(n_estimators=n_est, max_features="sqrt",
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=seed, oob_score=True, bootstrap=True)
    m.fit(X, y)
    return 1.0 - m.oob_score_


def tune(axis):
    expr, samples, y = discovery_samples(axis)
    genes = top_hvg(expr, samples, N_HVG)
    X = expr.loc[genes, samples].T.values
    n = len(y)
    resolution = 1.0 / n

    print("\n" + "=" * 76)
    print(f"AXIS: {axis}   {len(genes)} HVGs x {n} samples "
          f"(case {int(y.sum())} / ctrl {int((y == 0).sum())})")
    print("=" * 76)
    print(f"  OOB error resolution = 1/{n} = {resolution:.3f} "
          f"({100*resolution:.1f} percentage points per sample)")
    print(f"  averaging {N_SEEDS} forests per grid point")

    rows = []
    for n_est in GRID:
        t0 = time.time()
        errs = np.array([oob_error(X, y, n_est, RS + s) for s in range(N_SEEDS)])
        rows.append({"axis": axis, "n_estimators": n_est,
                     "oob_error_mean": round(float(errs.mean()), 5),
                     "oob_error_sd": round(float(errs.std(ddof=1)), 5),
                     "oob_error_se": round(float(errs.std(ddof=1) / np.sqrt(N_SEEDS)), 5),
                     "oob_error_min": round(float(errs.min()), 5),
                     "oob_error_max": round(float(errs.max()), 5),
                     "oob_accuracy_mean": round(1 - float(errs.mean()), 5),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  n={n_est:<5d} OOB error = {errs.mean():.4f} +/- {errs.std(ddof=1):.4f} "
              f"(min {errs.min():.4f}, max {errs.max():.4f})  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)

    # --- selection: lowest mean OOB error ---
    i_min = int(df["oob_error_mean"].idxmin())
    best = df.loc[i_min]
    rec = int(best["n_estimators"])

    # --- parsimonious alternative: smallest n within 1 SE of the minimum ---
    thr = float(best["oob_error_mean"] + best["oob_error_se"])
    within = df[df["oob_error_mean"] <= thr]
    parsimonious = int(within["n_estimators"].iloc[0]) if len(within) else rec

    spread = float(df["oob_error_mean"].max() - df["oob_error_mean"].min())

    print(f"\n  LOWEST OOB error: n_estimators = {rec} "
          f"(OOB error {best['oob_error_mean']:.4f}, "
          f"accuracy {1-best['oob_error_mean']:.4f})")
    print(f"  within 1 SE of the minimum: smallest n = {parsimonious}")
    print(f"  spread across the whole grid = {spread:.4f}")
    if spread < resolution:
        print(f"  [WARN] that spread is smaller than one OOB step ({resolution:.3f}).")
        print(f"         Differences between grid values are below the resolution of")
        print(f"         the metric, so the ranking of tree counts is not meaningful")
        print(f"         here. Any value on the plateau is defensible; prefer the")
        print(f"         parsimonious one and state that OOB was flat.")
    df["recommended"] = df["n_estimators"] == rec
    df["parsimonious"] = df["n_estimators"] == parsimonious
    df["oob_resolution"] = round(resolution, 5)
    return df, rec, parsimonious, spread < resolution

def make_figure(all_df, rec, pars, flat):
    """
    Publication-quality OOB error plot.
    Generates high-resolution PNG (600 dpi) and vector PDF.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[skip figure] matplotlib unavailable: {exc}")
        return


    df = pd.concat(all_df, ignore_index=True)

    axes_present = [a for a in AXES if a in set(df["axis"])]

    colors = {
        axes_present[0]: "#1f77b4"
    }

    if len(axes_present) > 1:
        colors[axes_present[1]] = "#d62728"


    # ============================
    # Figure configuration
    # ============================

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 600
    })


    fig, axs = plt.subplots(
        1,
        len(axes_present)+1,
        figsize=(18,6),
        constrained_layout=True
    )


    if len(axes_present)+1 == 1:
        axs=[axs]


    # =====================================
    # Individual axis plots
    # =====================================

    for i, axis in enumerate(axes_present):

        d = df[df["axis"] == axis].sort_values("n_estimators")

        ax = axs[i]

        col = colors[axis]


        minimum = d["oob_error_mean"].min()

        resolution = float(
            d["oob_resolution"].iloc[0]
        )


        # Resolution region
        ax.axhspan(
            minimum,
            minimum + resolution,
            alpha=0.15,
            color=col,
            label=f"Resolution band (1/n={resolution:.3f})"
        )


        # Main curve
        ax.errorbar(
            d["n_estimators"],
            d["oob_error_mean"],
            yerr=d["oob_error_sd"],
            marker="o",
            markersize=9,
            linewidth=3,
            capsize=5,
            color=col,
            label="Mean OOB ± SD"
        )


        # Seed variation
        ax.fill_between(
            d["n_estimators"],
            d["oob_error_min"],
            d["oob_error_max"],
            alpha=0.12,
            color=col,
            label="Seed range"
        )


        # Selected point

        best_n = int(rec[axis])

        best_val = float(
            d.loc[
                d["n_estimators"]==best_n,
                "oob_error_mean"
            ].iloc[0]
        )


        ax.scatter(
            best_n,
            best_val,
            s=180,
            marker="*",
            color="black",
            zorder=10,
            label=f"Selected {best_n} trees"
        )


        # Axis formatting

        ax.set_xscale("log")

        ax.set_xticks(
            d["n_estimators"]
        )

        ax.set_xticklabels(
            [str(int(x)) for x in d["n_estimators"]],
            rotation=45
        )


        ax.set_xlabel(
            "Number of trees"
        )

        ax.set_ylabel(
            "OOB error"
        )


        ax.set_title(
            f"{axis.upper()} axis",
            fontweight="bold"
        )


        ax.grid(
            linestyle="--",
            alpha=0.35
        )


        ax.legend(
            loc="best",
            frameon=True
        )



    # =====================================
    # Combined comparison plot
    # =====================================

    ax = axs[-1]


    for axis in axes_present:

        d = df[df["axis"]==axis].sort_values(
            "n_estimators"
        )

        ax.plot(
            d["n_estimators"],
            d["oob_error_mean"],
            marker="o",
            markersize=8,
            linewidth=3,
            label=axis.upper(),
            color=colors[axis]
        )


        ax.fill_between(
            d["n_estimators"],
            d["oob_error_mean"]-d["oob_error_sd"],
            d["oob_error_mean"]+d["oob_error_sd"],
            alpha=0.15,
            color=colors[axis]
        )


    ax.set_xscale("log")


    ticks = sorted(
        df["n_estimators"].unique()
    )

    ax.set_xticks(ticks)

    ax.set_xticklabels(
        [str(int(x)) for x in ticks],
        rotation=45
    )


    ax.set_xlabel(
        "Number of trees"
    )

    ax.set_ylabel(
        "OOB error"
    )


    ax.set_title(
        "Comparison of all axes",
        fontweight="bold"
    )


    ax.grid(
        linestyle="--",
        alpha=0.35
    )


    ax.legend()



    # Overall title

    fig.suptitle(
        "Random Forest OOB Error vs Number of Estimators",
        fontsize=18,
        fontweight="bold"
    )



    # Save high quality

    for ext in ["png","pdf"]:

        fig.savefig(
            QC_DIR /
            f"oob_error_vs_n_estimators_highres.{ext}",
            dpi=600,
            bbox_inches="tight"
        )


    plt.close(fig)


    print(
        f"[figure saved] {QC_DIR}/oob_error_vs_n_estimators_highres.png"
    )
def main():
    print("=" * 76)
    print("STEP 03b — n_estimators by OUT-OF-BAG error (read only)")
    print("=" * 76)
    print(f"  grid: {GRID}")
    print(f"  rule: lowest mean OOB error over {N_SEEDS} seeds")

    all_df, rec, pars, flat = [], {}, {}, {}
    for axis in AXES:
        df, r, p, f = tune(axis)
        all_df.append(df); rec[axis] = r; pars[axis] = p; flat[axis] = f

    out = pd.concat(all_df, ignore_index=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(QC_DIR / "rf_oob_tuning.tsv", sep="\t", index=False)
    make_figure(all_df, rec, pars, flat)

    print("\n" + "=" * 76)
    print("PASTE THIS INTO config/v2_config.yaml UNDER  rf:")
    print("=" * 76)
    print("  n_estimators_by_axis:")
    for axis in AXES:
        print(f"    {axis}: {rec[axis]}")
    for axis in AXES:
        if flat[axis]:
            print(f"  # {axis}: OOB was flat below its resolution; "
                  f"parsimonious alternative = {pars[axis]}")
    print(f"\n[SAVED] {QC_DIR / 'rf_oob_tuning.tsv'}")
    print("[DONE] nothing was modified; step 04 uses these after you edit the config.")


if __name__ == "__main__":
    main()