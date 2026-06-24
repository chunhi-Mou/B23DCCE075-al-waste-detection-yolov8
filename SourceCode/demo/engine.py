"""Model discovery"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("DEMO_RESULTS_DIR", REPO_ROOT / "results"))
BASELINE_FALLBACK_DIRS = (REPO_ROOT / "runs" / "baseline",)

STRATEGY_LABELS = {"random": "Random", "uncertainty": "Uncertainty",
                   "coreset": "CoreSet", "ppal": "PPAL"}
STRATEGY_ORDER = ("random", "uncertainty", "coreset", "ppal")
PREFERRED_SEED = 13

# AL budget schedule (configs/benchmark.yaml): init 5% + 2.5%/round, cap 20%.
_AL_INIT_PCT, _AL_STEP_PCT, _AL_CAP_PCT = 5.0, 2.5, 20.0
_AL_NAME_RE = re.compile(r"^(?P<strategy>[a-z]+)_s(?P<seed>\d+)_r(?P<round>\d+)$")
_BASELINE_NAME_RE = re.compile(r"^baseline_frac(?P<frac>[\d.]+)_seed(?P<seed>\d+)$")


@dataclass
class ModelInfo:
    label: str
    path: Path
    kind: str                # "baseline" | "al"
    strategy: str | None
    seed: int | None
    round: int | None
    data_pct: float | None
    map50: float | None


def _best_map50(run_dir: Path) -> float | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    vals: list[float] = []
    try:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                raw = row.get("metrics/mAP50(B)")
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return None
    return max(vals) if vals else None


def _al_data_pct(round_idx: int) -> float:
    return min(_AL_INIT_PCT + _AL_STEP_PCT * round_idx, _AL_CAP_PCT)


def _find_baseline() -> ModelInfo | None:
    roots = [RESULTS_DIR / "Baseline" / "runs" / "baseline", *BASELINE_FALLBACK_DIRS]
    candidates: list[tuple[float, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in root.iterdir():
            if not (run_dir / "weights" / "best.pt").exists():
                continue
            m = _BASELINE_NAME_RE.match(run_dir.name)
            candidates.append((float(m.group("frac")) if m else 1.0, run_dir))
    if not candidates:
        return None
    frac, run_dir = max(candidates, key=lambda t: t[0])  # the 100% Oracle
    base = baseline_summary()
    return ModelInfo(
        label=f"Baseline · {round(frac * 100)}% dữ liệu (Oracle)",
        path=run_dir / "weights" / "best.pt", kind="baseline", strategy=None,
        seed=None, round=None, data_pct=frac * 100,
        map50=(base or {}).get("test_mAP50") or _best_map50(run_dir),
    )


def _scan_al_runs() -> list[ModelInfo]:
    idx = _al_test_index()
    out: list[ModelInfo] = []
    for best in sorted(RESULTS_DIR.glob("*/runs/al/*/weights/best.pt")):
        run_dir = best.parent.parent
        m = _AL_NAME_RE.match(run_dir.name)
        if not m:
            continue
        rnd, seed, strat = int(m.group("round")), int(m.group("seed")), m.group("strategy")
        out.append(ModelInfo(
            label=run_dir.name, path=best, kind="al",
            strategy=strat, seed=seed, round=rnd,
            data_pct=_al_data_pct(rnd),
            map50=idx.get((strat, seed, rnd)) or _best_map50(run_dir),
        ))
    return out


def discover_models() -> dict[str, ModelInfo]:
    """Curated list: baseline + the final-round model (preferred seed) per strategy."""
    curated: dict[str, ModelInfo] = {}
    baseline = _find_baseline()
    if baseline is not None:
        curated[baseline.label] = baseline

    by_strategy: dict[str, list[ModelInfo]] = {}
    for info in _scan_al_runs():
        by_strategy.setdefault(info.strategy or "", []).append(info)

    for strategy in STRATEGY_ORDER:
        runs = by_strategy.get(strategy)
        if not runs:
            continue
        seeds = {r.seed for r in runs}
        seed = PREFERRED_SEED if PREFERRED_SEED in seeds else min(seeds)
        chosen = max((r for r in runs if r.seed == seed), key=lambda r: r.round or 0)
        name = STRATEGY_LABELS.get(strategy, strategy.title())
        label = f"{name} (AL) · vòng cuối ~{round(chosen.data_pct or 0)}% dữ liệu"
        curated[label] = ModelInfo(**{**chosen.__dict__, "label": label})
    return curated


def discover_all_models() -> dict[str, ModelInfo]:
    """Full list for the Dev tab: baseline + every AL run (all seeds & rounds)."""
    out: dict[str, ModelInfo] = {}
    baseline = _find_baseline()
    if baseline is not None:
        out[baseline.label] = baseline
    for info in _scan_al_runs():
        name = STRATEGY_LABELS.get(info.strategy or "", (info.strategy or "").title())
        label = f"{name} · seed {info.seed} · vòng {info.round} (~{round(info.data_pct or 0)}%)"
        out[label] = ModelInfo(**{**info.__dict__, "label": label})
    return out


@dataclass
class PredictResult:
    annotated: np.ndarray
    detections: list[dict]
    counts: dict[str, int]
    ms: float


_MODEL_CACHE: dict[str, YOLO] = {}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path: Path) -> YOLO:
    key = str(Path(path).resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key]


def predict(model: YOLO, image: np.ndarray, conf: float, iou: float) -> PredictResult:
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 4:        
        img = img[:, :, :3]
    bgr = np.ascontiguousarray(img[:, :, ::-1]) if img.ndim == 3 else img

    start = time.perf_counter()
    results = model.predict(bgr, conf=conf, iou=iou, device=_device(), verbose=False)
    ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    annotated = result.plot()[:, :, ::-1].copy() 
    names = result.names
    detections: list[dict] = []
    counts: dict[str, int] = {}
    for box in result.boxes:
        cls_name = names[int(box.cls.item())]
        detections.append({
            "cls_name": cls_name,
            "conf": float(box.conf.item()),
            "xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return PredictResult(annotated, detections, counts, ms)


CLASSES = ("cardboard", "paper", "glass", "metal", "plastic")


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_AL_ROWS: list[dict] | None = None


def al_results() -> list[dict]:
    """Per (strategy, seed, round) rows from every results/*/reports/al/results.csv."""
    global _AL_ROWS
    if _AL_ROWS is None:
        rows: list[dict] = []
        for p in sorted(RESULTS_DIR.glob("*/reports/al/results.csv")):
            try:
                with p.open(newline="") as fh:
                    for r in csv.DictReader(fh):
                        rows.append({
                            "strategy": r.get("strategy", ""),
                            "seed": int(r["seed"]),
                            "round": int(r["round"]),
                            "frac": _f(r.get("frac")),
                            "val_mAP50": _f(r.get("val_mAP50")),
                            "test_mAP50": _f(r.get("test_mAP50")),
                            "test_mAP50_95": _f(r.get("test_mAP50_95")),
                            **{c: _f(r.get(f"AP_{c}")) for c in CLASSES},
                        })
            except (OSError, KeyError, ValueError):
                continue
        _AL_ROWS = rows
    return _AL_ROWS


def _al_test_index() -> dict[tuple[str, int, int], float | None]:
    return {(r["strategy"], r["seed"], r["round"]): r["test_mAP50"] for r in al_results()}


def aubc_table() -> list[dict]:
    """Rows from every tbl_aubc.csv (AUBC, mAP at the 20% cap, n_seeds)."""
    out: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*/reports/al/tbl_aubc.csv")):
        try:
            with p.open(newline="") as fh:
                out.extend(csv.DictReader(fh))
        except OSError:
            continue
    return out


_BASELINE_TEST_RE = re.compile(r"Test mAP50\*\*:\s*([\d.]+)")
_BASELINE_AP_RE = re.compile(r"-\s*\*\*(\w+)\*\*:\s*([\d.]+)")


def baseline_summary() -> dict | None:
    """Oracle test mAP50 + per-class AP, parsed from the baseline SUMMARY.md."""
    for p in sorted(RESULTS_DIR.glob("Baseline/reports/baseline/*/SUMMARY.md")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = _BASELINE_TEST_RE.search(text)
        if not m:
            continue
        per_class: dict[str, float] = {}
        if "Per-class AP@50 (test)" in text:
            tail = text.split("Per-class AP@50 (test)", 1)[1]
            per_class = {n: float(v) for n, v in _BASELINE_AP_RE.findall(tail)
                         if n in CLASSES}
        return {"test_mAP50": float(m.group(1)), "per_class": per_class}
    return None