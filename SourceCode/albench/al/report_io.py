"""Read layer for the AL report: parses results.csv into typed
per-(strategy, seed) rounds + a coverage grid, sums confusion .npy."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NAMES = ["cardboard", "paper", "glass", "metal", "plastic"]
_AP = [f"AP_{n}" for n in NAMES]
_NL = [f"nlab_{n}" for n in NAMES]


def schedule_fracs(sch: dict) -> list[float]:
    n = round((sch["cap_frac"] - sch["init_frac"]) / sch["step_frac"])
    return [round(sch["init_frac"] + i * sch["step_frac"], 6)
            for i in range(n + 1)]             # [0]=init, [-1]=cap (R=len-1)


@dataclass
class Results:
    strategies: list
    seeds: list
    by: dict                                   # (strategy, seed) -> [Round]
    coverage: list                             # [{strategy,seed,n_rounds,...}]
    names: list


def load_results(csv_path: str) -> Results:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)
    by: dict = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["strategy"], int(row["seed"]))
            rec = {
                "round": int(row["round"]),
                "n_labeled": int(row["n_labeled"]),
                "frac": float(row["frac"]),
                "val_mAP50": float(row["val_mAP50"]),
                "test_mAP50": float(row["test_mAP50"]),
                "test_mAP50_95": float(row["test_mAP50_95"]),
                "ap": [float(row[c]) for c in _AP],
                "nlab": [int(float(row[c])) for c in _NL],
                "picks_sha": row["picks_sha"],
                "train_s": float(row["train_s"]),
            }
            by.setdefault(key, []).append(rec)
    for k in by:
        by[k].sort(key=lambda d: d["round"])
    strategies = sorted({s for s, _ in by})
    seeds = sorted({sd for _, sd in by})
    coverage = [
        {"strategy": s, "seed": sd, "n_rounds": len(by[(s, sd)]),
         "max_round": by[(s, sd)][-1]["round"]}
        for s in strategies for sd in seeds if (s, sd) in by]
    return Results(strategies, seeds, by, coverage, list(NAMES))


def load_confusion(confusion_dir: str, strategy: str, seeds: list):
    acc, used = None, []
    for sd in seeds:
        fp = Path(confusion_dir) / f"confusion_{strategy}_s{sd}.npy"
        if fp.exists():
            m = np.load(fp).astype(float)
            if acc is not None and m.shape != acc.shape:
                raise ValueError(f"{fp}: shape {m.shape} != {acc.shape}")
            acc = m if acc is None else acc + m
            used.append(sd)
    return acc, used
