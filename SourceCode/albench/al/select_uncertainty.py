"""S1 Uncertainty sampling (Brust et al., VISAPP 2019, arXiv:1809.09875):
squared 1-vs-2 margin (Eq.1) + max/sum/avg image aggregation (Eqs 2-4)."""
import numpy as np


def box_margin(p: np.ndarray) -> float:
    p = np.sort(np.asarray(p, dtype=float))[::-1]
    p1, p2 = (p[0], p[1]) if p.size >= 2 else (p[0], 0.0)
    return float((1.0 - (p1 - p2)) ** 2)   # Brust Eq.1 squared margin


def image_score(values, aggregation: str, empty_value: float) -> float:
    if not len(values):
        return float(empty_value)
    v = np.asarray(values, dtype=float)
    if aggregation == "max":
        return float(v.max())
    if aggregation == "sum":
        return float(v.sum())
    if aggregation == "avg":
        return float(v.mean())
    raise ValueError(f"unknown aggregation {aggregation!r}")


def select(model, candidates, labeled, budget, cfg, rng):
    if budget <= 0:
        return []
    from .score_predictor import ScorePredictor
    from .device import device
    u = cfg["al"]["uncertainty"]
    imgsz = cfg["al"]["imgsz"]
    bs = cfg["al"]["acq_batch"]
    pool = sorted(set(candidates) - set(labeled))
    scored = []
    # bound the source list (Ultralytics eagerly converts list sources)
    for start in range(0, len(pool), bs):
        paths = pool[start:start + bs]
        # rect=False => square imgsz => scores batch-invariant
        results = model.predict(
            paths, predictor=ScorePredictor, conf=u["conf"], imgsz=imgsz,
            device=device(), rect=False, batch=bs, stream=True, verbose=False)
        for path, res in zip(paths, results):
            vals = [box_margin(row.detach().cpu().numpy().astype(float))
                    for row in res.cls_scores]
            scored.append((image_score(vals, u["aggregation"],
                                       u["empty_value"]), path))
    scored.sort(key=lambda t: (-t[0], t[1]))             # tie-break by path
    return sorted(p for _, p in scored[:budget])
