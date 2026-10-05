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
def make_figure(all_df, rec, pars=None, flat=None):
    """
    PLOS ONE–compliant multi-panel OOB error figure.

    Panel titles (exact):
      - Severity axis
      - Warning sign axis
      - Comparison between axes

    Legend:
      - Inside top-left corner of each axes (minimal padding)
      - As small and compact as possible (7 pt)
      - Clear, short labels

    Resolution band:
      - Visually distinct light-green horizontal band
      - Legend explicitly states the resolution value

    Technical (PLOS One rules):
      - Physical size: 7.5 in × 4.0 in (max allowed width)
      - 300 dpi → 2250 × 1200 pixels (exactly at the upper limit)
      - Single-layer (flattened) TIFF with LZW compression
      - Arial (or Liberation Sans / DejaVu Sans fallback)
      - RGB, no transparency issues in EPS
      - No overall title / caption / figure number inside the figure
      - Exports: TIFF, EPS, PNG, JPEG, SVG, PDF
      - Output files named Fig3.*
    """

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker

    # ============================================================
    # Combine data
    # ============================================================
    try:
        df = pd.concat(all_df, ignore_index=True)
    except Exception as exc:
        print(f"[skip figure] could not combine input dataframes: {exc}")
        return

    axes_present = [a for a in AXES if a in set(df["axis"])]
    if not axes_present:
        print("[skip figure] no matching axes found")
        return

    # ============================================================
    # Font selection (PLOS prefers Arial)
    # ============================================================
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" in available_fonts:
        base_font = "Arial"
    elif "Liberation Sans" in available_fonts:
        base_font = "Liberation Sans"
    else:
        base_font = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": base_font,
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,

        "figure.dpi": 300,
        "savefig.dpi": 300,

        "pdf.fonttype": 42,          # TrueType – editable text in PDF/EPS
        "ps.fonttype": 42,
    })

    # ============================================================
    # Colors
    # ============================================================
    base_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    colors = {
        axis: base_colors[i % len(base_colors)]
        for i, axis in enumerate(axes_present)
    }

    # ============================================================
    # Helpers
    # ============================================================
    def blend_with_white(hex_color, alpha):
        """Pre-blend with white so EPS (no real transparency) looks correct."""
        rgb = matplotlib.colors.to_rgb(hex_color)
        return tuple(c * alpha + (1.0 - alpha) for c in rgb)

    def panel_title(axis_name):
        """Exact titles requested by user."""
        axis_upper = str(axis_name).upper()
        if axis_upper == "SEVERITY":
            return "Severity axis"
        if axis_upper in ("LEAKAGE", "WARNING"):
            return "Warning sign axis"
        return f"{axis_upper} axis"

    def nice_step(value):
        candidates = np.array([
            0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
            0.040, 0.050, 0.075, 0.100, 0.150, 0.200,
            0.250, 0.500
        ])
        valid = candidates[candidates >= value]
        if len(valid) > 0:
            return float(valid[0])
        return float(np.ceil(value / 0.1) * 0.1)

    def set_square_grid(ax, y_low, y_high, n_x_points):
        """Make plotting area square with matching number of major intervals."""
        n_intervals = max(1, n_x_points - 1)
        y_low = float(y_low)
        y_high = float(y_high)
        span = max(y_high - y_low, 0.01)

        pad = max(0.004, 0.04 * span)
        raw_low = y_low - pad
        raw_high = y_high + pad

        step = nice_step((raw_high - raw_low) / n_intervals)
        ymin = np.floor(raw_low / step) * step

        if (raw_high - ymin) / n_intervals > step:
            step = nice_step((raw_high - ymin) / n_intervals)
            ymin = np.floor(raw_low / step) * step

        ymax = ymin + n_intervals * step
        if ymax < raw_high:
            step = (raw_high - ymin) / n_intervals
            ymax = ymin + n_intervals * step

        yticks = np.linspace(ymin, ymax, n_intervals + 1)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.set_xlim(-0.25, n_x_points - 1 + 0.25)
        ax.set_box_aspect(1)

    # ============================================================
    # Shared X-axis positions
    # ============================================================
    grid_vals = sorted(df["n_estimators"].unique())
    pos = {v: i for i, v in enumerate(grid_vals)}
    x_pos = np.arange(len(grid_vals))

    # ============================================================
    # Figure layout
    # 7.5 in wide × 4.0 in high @ 300 dpi = 2250 × 1200 px
    # (exactly at the PLOS upper width limit)
    # ============================================================
    n_panels = len(axes_present) + 1
    fig, axs = plt.subplots(
        1, n_panels,
        figsize=(7.5, 4.0),
        constrained_layout=False
    )
    if n_panels == 1:
        axs = [axs]

    fig.subplots_adjust(
        left=0.08, right=0.99,
        bottom=0.16, top=0.88,
        wspace=0.38
    )

    # ============================================================
    # Individual axis panels
    # ============================================================
    for i, axis in enumerate(axes_present):
        d = df[df["axis"] == axis].sort_values("n_estimators")
        x = d["n_estimators"].map(pos).to_numpy()
        ax = axs[i]
        col = colors[axis]

        minimum = float(d["oob_error_mean"].min())
        resolution = float(d["oob_resolution"].iloc[0])

        # ---- Resolution band (clear light-green) ----
        h_res = ax.axhspan(
            minimum, minimum + resolution,
            color=blend_with_white("#4caf50", 0.28),
            zorder=0, linewidth=0
        )

        # ---- Seed range ----
        h_seed = ax.fill_between(
            x, d["oob_error_min"], d["oob_error_max"],
            color=blend_with_white(col, 0.18),
            zorder=1, linewidth=0
        )

        # ---- Mean OOB ± SD ----
        h_err = ax.errorbar(
            x, d["oob_error_mean"], yerr=d["oob_error_sd"],
            marker="o", markersize=3.6,
            linewidth=1.3, capsize=2,
            color=col, zorder=3
        )

        # ---- Selected tree count ----
        best_n = int(rec[axis])
        best_val = float(
            d.loc[d["n_estimators"] == best_n, "oob_error_mean"].iloc[0]
        )
        h_sel = ax.scatter(
            pos[best_n], best_val,
            s=48, marker="*", color="black", zorder=5
        )

        # X ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(int(v)) for v in grid_vals],
            rotation=45, ha="right"
        )

        # Square grid
        y_low = min(float(d["oob_error_min"].min()), minimum)
        y_high = max(float(d["oob_error_max"].max()), minimum + resolution)
        set_square_grid(ax, y_low, y_high, len(grid_vals))

        ax.set_xlabel("Number of trees")
        ax.set_ylabel("OOB error")
        ax.set_title(panel_title(axis), fontweight="bold", pad=5)

        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)

        # --------------------------------------------------------
        # Legend – flush top-left corner, minimal size
        # --------------------------------------------------------
        ax.legend(
            [h_res, h_seed, h_sel, h_err],
            [
                f"Resolution band (1/n={resolution:.3f})",
                "Seed range",
                f"Selected {best_n} trees",
                "Mean OOB ± SD"
            ],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.12,
            frameon=True,
            fancybox=False,
            edgecolor="0.55",
            framealpha=0.92,
            borderpad=0.25,
            labelspacing=0.18,
            handlelength=1.5,
            handletextpad=0.4,
            fontsize=7
        )

    # ============================================================
    # Comparison panel
    # ============================================================
    ax = axs[-1]
    line_handles = []
    line_labels = []
    ymins, ymaxs = [], []

    for axis in axes_present:
        d = df[df["axis"] == axis].sort_values("n_estimators")
        x = d["n_estimators"].map(pos).to_numpy()

        line, = ax.plot(
            x, d["oob_error_mean"],
            marker="o", markersize=3.6, linewidth=1.3,
            color=colors[axis], zorder=3
        )

        low  = d["oob_error_mean"] - d["oob_error_sd"]
        high = d["oob_error_mean"] + d["oob_error_sd"]
        ax.fill_between(
            x, low, high,
            color=blend_with_white(colors[axis], 0.15),
            zorder=1, linewidth=0
        )

        ymins.append(float(low.min()))
        ymaxs.append(float(high.max()))
        line_handles.append(line)

        axis_upper = str(axis).upper()
        if axis_upper in ("LEAKAGE", "WARNING"):
            line_labels.append("WARNING")
        else:
            line_labels.append("SEVERITY")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(int(v)) for v in grid_vals],
        rotation=45, ha="right"
    )

    set_square_grid(ax, min(ymins), max(ymaxs), len(grid_vals))

    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error")
    ax.set_title("Comparison between axes", fontweight="bold", pad=5)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)

    # Legend – flush top-left corner
    ax.legend(
        line_handles, line_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.12,
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        framealpha=0.92,
        borderpad=0.25,
        labelspacing=0.18,
        handlelength=1.5,
        handletextpad=0.4,
        fontsize=7,
        ncol=1
    )

    # ============================================================
    # Save outputs (all PLOS-compatible formats) – named Fig3
    # ============================================================
    out_base = QC_DIR / "Fig3"

    # TIFF – single layer, LZW compression (PLOS requirement)
    fig.savefig(
        out_base.with_suffix(".tif"),
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    # EPS
    fig.savefig(
        out_base.with_suffix(".eps"),
        format="eps",
        bbox_inches="tight",
        pad_inches=0.02
    )

    # PNG
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    # JPEG
    fig.savefig(
        out_base.with_suffix(".jpeg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"quality": 95}
    )

    # SVG
    fig.savefig(
        out_base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.02
    )

    # PDF
    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close(fig)

    for ext in ["tif", "eps", "png", "jpeg", "svg", "pdf"]:
        print(f"[figure saved] {out_base}.{ext}")
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
    print(f"\n[SAVED] -> {OUT}\n[DONE]")

if __name__ == "__main__":
    main()