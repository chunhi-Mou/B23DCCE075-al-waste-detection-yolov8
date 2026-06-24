"""One (strategy, seed) AL run."""
import csv
import hashlib
import random
from pathlib import Path

from . import state
from .dataset import pool_images, image_classes, write_round_data_yaml
from .device import device
from .init_set import stratified_initial
from .select_random import select as _sel_random
from .select_uncertainty import select as _sel_uncertainty
from .select_coreset import select as _sel_coreset
from .select_ppal import select as _sel_ppal

SELECTORS = {"random": _sel_random, "uncertainty": _sel_uncertainty,
             "coreset": _sel_coreset, "ppal": _sel_ppal}

_CSV_COLS = ["strategy", "seed", "round", "n_labeled", "frac",
             "val_mAP50", "test_mAP50", "test_mAP50_95",
             "AP_cardboard", "AP_paper", "AP_glass", "AP_metal", "AP_plastic",
             "nlab_cardboard", "nlab_paper", "nlab_glass", "nlab_metal",
             "nlab_plastic", "picks_sha", "train_s"]


def _pool_items(data_yaml):
    return [(p, image_classes(p)) for p in pool_images(data_yaml)]


def _csv_has_row(csv_path: str, strategy: str, seed: int, r: int) -> bool:
    """True if results.csv already has this (strategy, seed, round) row."""
    p = Path(csv_path)
    if not p.exists():
        return False
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("strategy") == strategy
                    and row.get("seed") == str(seed)
                    and row.get("round") == str(r)):
                return True
    return False


def _rounds(sch):
    fracs, f = [], sch["init_frac"]
    while f <= sch["cap_frac"] + 1e-9:
        fracs.append(round(f, 6))
        f += sch["step_frac"]
    return fracs                       # [0]=init, [-1]=cap


def _sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _train_round(strategy, seed, r, data_yaml, cfg):
    import time
    from ultralytics import YOLO
    from albench.repro import set_seed
    al = cfg["al"]
    set_seed(seed)
    t0 = time.time()
    model = YOLO(al["model"])
    name = f"{strategy}_s{seed}_r{r}"

    _ema = None
    if strategy == "ppal":
        from .ppal_difficulty import (QualityEMA, make_batch_quality_callbacks)
        pp = al["ppal"]
        _ema = QualityEMA(cfg["classes"]["nc"], pp["ema_momentum"])
        for event, fn in make_batch_quality_callbacks(
                _ema, pp["xi"], pp["score_thr"]).items():
            model.add_callback(event, fn)

    proj = str(Path(al["project"]).resolve())
    model.train(data=data_yaml, epochs=al["epochs"], imgsz=al["imgsz"],
                batch=al["batch"], cache=al.get("cache", False),
                patience=al["patience"], device=device(),
                project=proj, name=name,
                pretrained=al.get("pretrained", True), seed=seed,
                plots=True, val=al.get("val", True), exist_ok=True)
    run_dir = Path(model.trainer.save_dir)
    last = run_dir / "weights" / "last.pt"
    model_val = YOLO(str(last))
    em = model_val.val(data=data_yaml, split="val", device=device(),
                       project=proj, name=f"{name}_val", plots=False,
                       exist_ok=True, verbose=False)
    tm = model_val.val(data=data_yaml, split="test", device=device(),
                       project=proj, name=f"{name}_test", plots=False,
                       exist_ok=True, verbose=False)
    ap = [float(x) for x in tm.box.maps[:5]]
    if _ema is not None:
        model_val._ppal_class_quality = _ema.value
    del model
    import gc
    gc.collect()
    return {"val_map50": float(em.box.map50),
            "test_map50": float(tm.box.map50),
            "test_map50_95": float(tm.box.map),
            "ap": ap, "train_s": time.time() - t0,
            "confusion": tm.confusion_matrix.matrix,   # raw counts
            "ppal_quality": (_ema.value if _ema is not None else None),
            "model": model_val}


def run(strategy, seed, cfg, data_yaml, *, csv_path=None,
        train_fn=_train_round, select_fns=SELECTORS):
    al = cfg["al"]
    sch = al["schedule"]
    csv_path = csv_path or str(Path(cfg["reports_dir"]) / "al" / "results.csv")
    splits = Path(cfg.get("splits_dir", "splits"))
    items = _pool_items(data_yaml)
    keys = [k for k, _ in items]
    fracs = _rounds(sch)
    R = len(fracs) - 1

    rng = random.Random(seed)
    labeled = set(stratified_initial(items, sch["init_frac"], rng))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_COLS)

    # every done round must have a CSV row
    _missing = [r for r in range(R + 1)
                if state.is_done(al["state_dir"], strategy, seed, r)
                and not _csv_has_row(csv_path, strategy, seed, r)]
    if _missing:
        raise RuntimeError(
            f"CSV/state inconsistency for {strategy} s{seed}: state says "
            f"rounds {_missing} are done but results.csv has no matching "
            f"row. Likely cause: a crash with the pre-fix write order. "
            f"Resolve by deleting the orphaned state file(s) and re-running, "
            f"or by restoring the missing CSV row(s) from the state file.")

    for r in range(R + 1):
        if state.is_done(al["state_dir"], strategy, seed, r):
            d = state.load(al["state_dir"], strategy, seed, r)
            labeled = set(d["next_labeled"])
            if "rng_state" in d:
                rng.setstate(state.decode_rng_state(d["rng_state"]))
            continue
        splits.mkdir(parents=True, exist_ok=True)
        txt = splits / f"al_{strategy}_s{seed}_r{r}.txt"
        txt.write_text("\n".join(sorted(labeled)) + "\n")
        ry = splits / f"data_al_{strategy}_s{seed}_r{r}.yaml"
        write_round_data_yaml(data_yaml, str(txt), str(ry))
        m = train_fn(strategy, seed, r, str(ry), cfg)
        if r == R and m.get("confusion") is not None:
            import numpy as np
            cm_path = Path(csv_path).parent / f"confusion_{strategy}_s{seed}.npy"
            np.save(cm_path, np.asarray(m["confusion"]))
        if strategy == "ppal" and m.get("ppal_quality") is not None:
            import numpy as np
            np.save(Path(csv_path).parent /
                    f"ppal_quality_{strategy}_s{seed}_r{r}.npy",
                    np.asarray(m["ppal_quality"]))
        if r < R:
            cands = sorted(set(keys) - labeled)
            nxt = max(1, round(len(keys) * fracs[r + 1]))
            picks = select_fns[strategy](
                m.get("model"), cands, labeled, nxt - len(labeled), cfg, rng)
            next_labeled = sorted(labeled | set(picks))
            _diag = getattr(m.get("model"), "_ppal_acq_diag", None)
            if strategy == "ppal" and _diag is not None:
                import numpy as np
                np.savez_compressed(
                    Path(csv_path).parent /
                    f"ppal_acq_{strategy}_s{seed}_r{r}.npz", **_diag)
        else:
            next_labeled = sorted(labeled)
        ap = m.get("ap", [0.0] * 5)
        nlab = _class_counts(items, labeled)
        # write the CSV row before marking state done
        if not _csv_has_row(csv_path, strategy, seed, r):
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    strategy, seed, r, len(labeled), fracs[r],
                    m["val_map50"], m["test_map50"],
                    m.get("test_map50_95", 0.0),
                    *ap, *nlab, _sha(labeled), m.get("train_s", 0.0)])
                f.flush()
                import os as _os
                _os.fsync(f.fileno())
        state.write_atomic(al["state_dir"], strategy, seed, r, {
            "labeled": sorted(labeled), "next_labeled": next_labeled,
            "rng_state": state.encode_rng_state(rng.getstate()),
            "val_map50": m["val_map50"], "test_map50": m["test_map50"]})
        labeled = set(next_labeled)


def _class_counts(items, labeled):
    cnt = [0, 0, 0, 0, 0]
    by = dict(items)
    for k in labeled:
        for c in by.get(k, frozenset()):
            if 0 <= c < 5:
                cnt[c] += 1
    return cnt
