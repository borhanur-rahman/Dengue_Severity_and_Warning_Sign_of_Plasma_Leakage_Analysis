#!/usr/bin/env python3
"""Step 01 — build sample metadata for both axes.

Your original script, with small robustness fixes only:
  * characteristics parsed as key:value (handles 'Cryopreserved PBMC' in a
    constant 'tissue' field, and any field-order differences between the two
    GSE178240 series matrices) — prevents the 0-PBMC-selected failure;
  * subject count reported (the independence check: 87 PBMC = 87 subjects);
  * title-uniqueness and merge-cardinality assertions.
No condition labels, platform assignments, or sample selection logic changed.
"""
from pathlib import Path
import gzip
import re
import urllib.request

import pandas as pd

from paths import CFG, seed_everything, ensure_data_dirs, META_DIR, metadata_path

seed_everything()
ensure_data_dirs()


gse178240_counts = Path(CFG["raw_data"]["gse178240_counts"])
gse215835_counts = Path(CFG["raw_data"]["gse215835_counts"])
SUBTYPE = CFG["axes"]["leakage"]["cell_subtype"]

PLATFORM_NAME = CFG["platform_names"]
from paths import SERIES_DIR
SERIES = {gpl: SERIES_DIR / fn for gpl, fn in CFG["raw_data"]["gse178240_series"].items()}
BASE = CFG["raw_data"]["gse178240_series_base_url"]
SERIES_URL = {gpl: BASE + fn for gpl, fn in CFG["raw_data"]["gse178240_series"].items()}


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", str(t).strip().lower()).strip("_")


def download_if_missing(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("[HAVE]", dest.name); return
    print("[DOWNLOAD]", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def parse_series(path, gpl):
    """Parse a GEO series matrix; characteristics split on first colon."""
    scalar, char_rows = {}, []
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0].replace("!Sample_", "")
            vals = [p.strip().strip('"') for p in parts[1:]]
            if key == "characteristics_ch1":
                char_rows.append(vals)
            else:
                scalar.setdefault(key, vals)
    accs = scalar.get("geo_accession", [])
    n = len(accs)
    df = pd.DataFrame({"gsm": accs, "title": scalar.get("title", [""] * n)[:n],
                       "platform_gpl": gpl, "platform": PLATFORM_NAME[gpl]})
    char_cols = {}
    for row in char_rows:
        row = (row + [""] * n)[:n]
        for i, cell in enumerate(row):
            if not cell or ":" not in cell:
                continue
            k, v = cell.split(":", 1)
            char_cols.setdefault("ch_" + slug(k), [""] * n)[i] = v.strip()
    for c, vals in char_cols.items():
        df[c] = vals
    return df


def find_subtype_col(df):
    ch = [c for c in df.columns if c.startswith("ch_")]
    for needle in ("cell_subtype", "subtype", "cell_type", "celltype"):
        for c in ch:
            if needle in c.lower():
                return c
    return None


def normalise_subtype(v):
    v = re.sub(r"(?i)\b(cryopreserved|frozen|fresh|sorted|cells?)\b", " ", str(v))
    v = re.sub(r"[+/ ]+", "_", v.strip())
    return re.sub(r"_+", "_", v).strip("_")


def warning_label(t):
    t = str(t).lower()
    if "without warning" in t: return "D-W"
    if "with warning" in t: return "D+W"
    return "Unknown"


def build_gse178240():
    print("\n" + "=" * 70 + f"\n[GSE178240] leakage metadata — subtype={SUBTYPE}\n" + "=" * 70)
    header = pd.read_csv(gse178240_counts, sep="\t", nrows=1).columns.tolist()
    all_cols = header[1:]
    print(f"[GSE178240] total count columns: {len(all_cols)}")

    frames = []
    for gpl, path in SERIES.items():
        download_if_missing(SERIES_URL[gpl], path)
        frames.append(parse_series(path, gpl))
    sm = pd.concat(frames, ignore_index=True)

    assert sm["title"].is_unique, "Duplicate titles in series matrix"

    warn_col = next((c for c in sm.columns if c.startswith("ch_")
                     and sm[c].astype(str).str.contains("warning", case=False, na=False).any()), None)
    subtype_col = find_subtype_col(sm)
    if warn_col is None or subtype_col is None:
        raise ValueError(f"Could not locate warning/subtype fields: "
                         f"{[c for c in sm.columns if c.startswith('ch_')]}")

    sm["true_label"] = sm[warn_col].map(warning_label)
    sm["cell_subtype"] = sm[subtype_col].map(normalise_subtype)

    joined = pd.DataFrame({"sample_id": all_cols}).merge(
        sm[["title", "gsm", "platform", "platform_gpl", "true_label", "cell_subtype"]],
        left_on="sample_id", right_on="title", how="left").drop(columns="title")
    assert len(joined) == len(all_cols), "merge changed row count"

    print("[GSE178240] cell subtypes in count columns:")
    print(joined["cell_subtype"].value_counts(dropna=False).to_string())

    want = normalise_subtype(SUBTYPE).upper()
    df = joined[joined["cell_subtype"].astype(str).str.upper() == want].copy().reset_index(drop=True)
    print(f"\n[GSE178240] '{SUBTYPE}' samples selected: {len(df)}")
    if df.empty:
        raise ValueError(f"No {SUBTYPE} columns. Available: "
                         f"{sorted(joined['cell_subtype'].dropna().unique())}")
    if df["gsm"].isna().any():
        raise ValueError("Unmatched samples: " +
                         str(df.loc[df["gsm"].isna(), "sample_id"].head().tolist()))
    if (df["true_label"] == "Unknown").any():
        raise ValueError("Unknown warning-sign label; check parsing.")

    # independence check (subject = title stripped of subtype token)
    def subj(t):
        s = str(t)
        for tok in sorted({want, *want.split("_")}, key=len, reverse=True):
            s = re.sub(rf"[_\-. ]*{re.escape(tok)}[_\-. ]*$", "", s, flags=re.I)
        return s.strip("_-. ")
    df["subject"] = df["sample_id"].map(subj)
    print(f"[GSE178240] unique subjects: {df['subject'].nunique()} | samples: {len(df)}")

    df["dataset"] = "GSE178240"
    df["axis"] = "leakage"
    df["condition"] = df["true_label"]
    df["case_control_role"] = df["condition"].map({"D+W": "case", "D-W": "control"})
    df = df[["sample_id", "gsm", "subject", "dataset", "axis", "condition",
             "case_control_role", "platform", "platform_gpl", "cell_subtype"]]
    df.to_csv(metadata_path("GSE178240"), sep="\t", index=False)

    print(f"\n[GSE178240] condition x platform:")
    print(pd.crosstab(df["condition"], df["platform"], margins=True))
    return df


def build_gse215835():
    print("\n" + "=" * 70 + "\n[GSE215835] severity metadata\n" + "=" * 70)
    header = pd.read_csv(gse215835_counts, sep="\t", nrows=1).columns.tolist()
    cols = [c for c in header if str(c).endswith("_DF") or str(c).endswith("_DHF")]
    print(f"[GSE215835] sample columns detected: {len(cols)}")
    if len(cols) != 11:
        raise ValueError(f"Expected 11 sample columns, found {len(cols)}")

    rows = []
    for s in cols:
        cond = "DHF" if str(s).endswith("_DHF") else "DF"
        rows.append({"sample_id": s, "gsm": "", "subject": s, "dataset": "GSE215835",
                     "axis": "severity", "condition": cond,
                     "case_control_role": "case" if cond == "DHF" else "control",
                     "platform": "single", "platform_gpl": "single", "cell_subtype": "PBMC"})
    df = pd.DataFrame(rows)
    df.to_csv(metadata_path("GSE215835"), sep="\t", index=False)
    print("[GSE215835] condition counts:")
    print(df["condition"].value_counts())
    return df


def main():
    lk = build_gse178240()
    sv = build_gse215835()
    merged = pd.concat([lk, sv], ignore_index=True)
    merged.to_csv(str(META_DIR / "merged_metadata.tsv"), sep="\t", index=False)
    print("\n" + "=" * 70 + "\nMERGED METADATA\n" + "=" * 70)
    print("shape:", merged.shape)
    print(pd.crosstab(merged["axis"], merged["condition"], margins=True))
    print("\n[DONE] Step 01 completed.")


if __name__ == "__main__":
    main()