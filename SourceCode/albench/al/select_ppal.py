"""S3 PPAL (Yang et al., CVPR 2024, arXiv:2211.11612). Two-stage selector:
Stage1 DCUS uncertainty + Stage2 CCMS diversity. Inference-only."""
import numpy as np

from .ppal_difficulty import reweight
from .ccms import image_distance_matrix, kmedoids


def _real_feature_fn(model, paths, cfg):
    from .ppal_features import extract_det_features
    return extract_det_features(model, paths, cfg)


def _box_entropy(s: float) -> float:
    # binary entropy  H(s) = -s log s - (1-s) log(1-s)
    s = float(np.clip(s, 1e-10, 1 - 1e-10))
    return -(s * np.log(s + 1e-10) + (1 - s) * np.log(1 - s + 1e-10))


def select(model, candidates, labeled, budget, cfg, rng, *,
           feature_fn=_real_feature_fn):
    if budget <= 0 or model is None:
        return []
    from .score_predictor import ScorePredictor
    from .device import device
    pp = cfg["al"]["ppal"]
    imgsz = cfg["al"]["imgsz"]
    pool = sorted(set(candidates) - set(labeled))
    cq = getattr(model, "_ppal_class_quality", None)
    if cq is None:
        raise RuntimeError("ppal: model has no _ppal_class_quality "
                           "(quality-EMA callback not run)")
    w = reweight(np.asarray(cq, float), pp["alpha"], pp["weight_ub"])

    # Stage 1: DCUS uncertainty
    # reset so predict() installs ScorePredictor
    model.predictor = None
    scored = []
    for path in pool:
        res = model.predict(path, predictor=ScorePredictor,
                            conf=pp["score_thr"], imgsz=imgsz,
                            device=device(), verbose=False)[0]
        tot = 0.0
        for row in res.cls_scores:
            p = np.asarray(row.detach().cpu().numpy()
                           if hasattr(row, "detach") else row, float)
            s = float(p.max())
            tot += _box_entropy(s) * float(w[int(p.argmax())])
        scored.append((tot, path))
    scored.sort(key=lambda t: (-t[0], t[1]))
    pool_n = min(pp["delta"] * budget, len(scored))
    cand = sorted(p for _, p in scored[:pool_n])

    # Stage 2: CCMS diversity
    feats, labels, scores = feature_fn(model, cand, cfg)
    D = image_distance_matrix(feats, labels, scores, pp["score_thr"])
    pick = kmedoids(D, min(budget, len(cand)), pp["kmeans_iter"], rng)
    # Acquisition diagnostics
    from .ppal_stage2 import acquisition_diag
    model._ppal_acq_diag = acquisition_diag(
        D, pick, np.array([t for t, _ in scored], float), pool_n)
    return sorted(cand[i] for i in pick)
