#!/usr/bin/env python3
"""Shared config + run-directory helper for the v2 pipeline.
Put in scripts/ next to the numbered steps.

Loads config/v2_config.yaml as CFG and seeds RNGs. Also manages the per-run
output directories so re-running a configuration archives (never overwrites)
the previous output.

PATH SAFETY: every script imports this module first, so anchoring the working
directory to the project root here makes all relative paths in the pipeline
(config, data/, v2/) resolve correctly no matter where the runner
(Code Ocean run.sh, cron, an IDE) invokes python from.
"""
from __future__ import annotations
import datetime as _dt
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

# --- anchor all relative paths to the project root (scripts/ is one level down)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

CONFIG_PATH = PROJECT_ROOT / "config" / "paths.yaml"
if not CONFIG_PATH.exists():
    sys.exit(f"Missing {CONFIG_PATH}. Copy v2_config.yaml into config/.")
CFG: dict = yaml.safe_load(CONFIG_PATH.read_text())

_o = CFG["output"]
V2 = Path(_o["root"])
META_DIR = Path(_o["metadata_dir"])
PROC_DIR = Path(_o["processed_dir"])
QC_DIR = Path(_o["qc_dir"])
SERIES_DIR = Path(_o["series_dir"])
RUNS = Path(_o["runs_dir"])
ARCHIVE = RUNS / "_archive"
LATEST = RUNS / "_latest.txt"
ARCHIVE_RUNS = bool(_o.get("archive_previous_runs", True))

_SEED = int(CFG.get("reproducibility", {}).get("seed", CFG["rf"]["random_state"]))


def seed_everything(seed: int = _SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[repro] tip: launch with  PYTHONHASHSEED=0 python ...")


def ensure_data_dirs() -> None:
    for d in (META_DIR, PROC_DIR, QC_DIR, SERIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def make_run_id() -> str:
    return "results"


def run_dir(run_id: str, step: str, config: dict | None = None) -> Path:
    d = RUNS / run_id / step
    if d.exists() and ARCHIVE_RUNS and any(d.iterdir()):
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = ARCHIVE / f"{run_id}__{step}__{ts}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
        print(f"  [archive] previous {step} preserved at {dest}")
    d.mkdir(parents=True, exist_ok=True)
    if config is not None:
        cfg = dict(config)
        cfg["_run_id"] = run_id; cfg["_step"] = step
        cfg["_written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        (d / "_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return d


def read_dir(run_id: str, step: str) -> Path:
    d = RUNS / run_id / step
    if not d.exists():
        avail = sorted(p.name for p in RUNS.glob("*") if p.is_dir() and p.name != "_archive")
        sys.exit(f"Missing {d}\n  Run the earlier step first. Runs: {avail or '(none)'}")
    return d


def set_latest(run_id: str) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(run_id + "\n")


def resolve(run_id: str | None) -> str:
    if run_id:
        print(f"  run id: {run_id} (explicit)"); return run_id
    if not LATEST.exists():
        sys.exit("No v2/runs/_latest.txt — run step 04 first, or set RUN_ID.")
    rid = LATEST.read_text().strip()
    print(f"  run id: {rid} (from _latest.txt)"); return rid


def dataset_tag(axis: str) -> str:
    return CFG["axes"][axis]["dataset"]


def counts_path(tag: str) -> Path:
    return PROC_DIR / f"{tag}_gene_counts.tsv.gz"


def logcpm_path(tag: str) -> Path:
    return PROC_DIR / f"{tag}_log2cpm.tsv.gz"


def metadata_path(tag: str) -> Path:
    return META_DIR / f"{tag}_metadata.tsv"