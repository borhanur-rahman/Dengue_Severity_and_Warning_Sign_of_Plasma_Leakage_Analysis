#!/usr/bin/env python3
"""Step 03 — CPM filter + log2, each dataset independently.

ONLY the low-expression filtering rule changed. Normalisation (CPM), the CPM
threshold, the log2(CPM+1) transform, the zero-library guard and the per-dataset
independence are all exactly as before.

Why the rule changed
--------------------
A fixed `min_samples = 5` means different things in the two cohorts: 5.7% of
GSE178240 (87 samples) but 45% of GSE215835 (11 samples) — an eightfold
difference in effective stringency that a reviewer will query. The new default
scales the requirement with each dataset's smallest condition group, so the
stringency is comparable across cohorts:

    filter_mode: group_fraction   (default)
        CPM >= min_cpm in >= ceil(min_group_fraction x smallest group) samples
        GSE178240: smallest group 33  ->  >= 17 samples
        GSE215835: smallest group  5  ->  >=  3 samples

    filter_mode: fixed            (your original behaviour, kept for comparison)
        CPM >= min_cpm in >= min_samples samples

Both rules are evaluated and printed every run so the impact is visible; only
the configured one is written to disk.

NOTE: this matrix is for feature selection and plotting. Differential
expression, if ever added, requires raw counts.

IMPORTANT: changing the filter changes the gene universe, so steps 4-7 must be
re-run after this step. Compare the new top-50 lists against the archived ones.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from paths import (CFG, seed_everything, ensure_data_dirs, counts_path,
                      logcpm_path, metadata_path)

seed_everything()
ensure_data_dirs()

_pre = CFG["preprocessing"]
MIN_CPM = float(_pre["min_cpm"])
FILTER_MODE = _pre.get("filter_mode", "fixed")
MIN_SAMPLES = int(_pre.get("min_samples", 5))
MIN_GROUP_FRACTION = float(_pre.get("min_group_fraction", 0.5))
MIN_SAMPLES_FLOOR = int(_pre.get("min_samples_floor", 2))


def load_counts(path):
    df = pd.read_csv(path, sep="\t", index_col=0, compression="infer")
    df.index = df.index.astype(str); df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def smallest_group(tag):
    """Size of the smallest condition group in this dataset's metadata."""
    m = pd.read_csv(metadata_path(tag), sep="\t")
    vc = m["condition"].value_counts()
    return int(vc.min()) if len(vc) else 0


def _group_rule(n_small):
    """max(floor, ceil(fraction x smallest group)).

    The floor keeps a minimum absolute number of expressing samples; the
    fraction makes the requirement scale with the study design. Whichever is
    larger binds, so no cohort is ever filtered more loosely than the floor.
    """
    frac = math.ceil(MIN_GROUP_FRACTION * n_small)
    k = max(MIN_SAMPLES_FLOOR, frac)
    binds = "floor" if MIN_SAMPLES_FLOOR >= frac else "group fraction"
    return (k, f"CPM>={MIN_CPM} in >={k} samples "
               f"= max(floor {MIN_SAMPLES_FLOOR}, "
               f"{MIN_GROUP_FRACTION}x smallest group {n_small} = {frac}) "
               f"[{binds} binds]")


def required_samples(tag, n_samples):
    """Return (k, description) for every rule, plus the active one."""
    n_small = smallest_group(tag)
    rules = {
        "fixed": (MIN_SAMPLES,
                  f"CPM>={MIN_CPM} in >={MIN_SAMPLES} samples "
                  f"({100*MIN_SAMPLES/n_samples:.1f}% of {n_samples})"),
        "group_fraction": _group_rule(n_small),
    }
    if FILTER_MODE not in rules:
        raise ValueError(f"unknown filter_mode '{FILTER_MODE}'; "
                         f"use one of {sorted(rules)}")
    return rules, n_small


def cpm_filter_log2(counts, tag):
    lib = counts.sum(axis=0)
    if (lib <= 0).any():
        raise ValueError(f"[{tag}] zero-library samples: {lib[lib <= 0].index.tolist()}")
    cpm = counts.div(lib, axis=1) * 1e6
    n_expressed = (cpm >= MIN_CPM).sum(axis=1)

    rules, n_small = required_samples(tag, counts.shape[1])

    # show every rule so the effect of the change is visible in the same run
    print(f"[{tag}] smallest condition group: {n_small} | samples: {counts.shape[1]}")
    for name, (k, desc) in rules.items():
        kept = int((n_expressed >= k).sum())
        mark = "  <== ACTIVE" if name == FILTER_MODE else ""
        print(f"    {name:<15} {desc:<62} -> {kept:>6} genes{mark}")

    k_active = rules[FILTER_MODE][0]
    keep = n_expressed >= k_active
    log2cpm = np.log2(cpm.loc[keep] + 1.0)
    print(f"[{tag}] genes {counts.shape[0]} -> {int(keep.sum())} kept "
          f"[{FILTER_MODE}] | samples: {counts.shape[1]}")
    return log2cpm


def process(counts_file, out_file, tag):
    print("\n" + "=" * 70 + f"\nNORMALIZE {tag} (within-dataset)\n" + "=" * 70)
    counts = load_counts(counts_file)
    log2cpm = cpm_filter_log2(counts, tag)
    out = out_file
    log2cpm.to_csv(out, sep="\t", compression="gzip")
    print(f"[{tag}] log2CPM saved: {log2cpm.shape} -> {out}")
    return log2cpm


def main():
    if FILTER_MODE == "group_fraction":
        print(f"[filter] rule = max(floor {MIN_SAMPLES_FLOOR}, "
              f"{MIN_GROUP_FRACTION} x smallest condition group), CPM >= {MIN_CPM}")
    else:
        print(f"[filter] rule = CPM >= {MIN_CPM} in >= {MIN_SAMPLES} samples [fixed]")
    a = process(counts_path("GSE178240"), logcpm_path("GSE178240"), "GSE178240")
    b = process(counts_path("GSE215835"), logcpm_path("GSE215835"), "GSE215835")

    common = len(set(a.index) & set(b.index))
    print("\n" + "=" * 70)
    print(f"cross-axis universe (genes measurable in both): {common}")
    print("=" * 70)
    print("\n[DONE] Step 03 completed. Each dataset normalized independently.")
  


if __name__ == "__main__":
    main()