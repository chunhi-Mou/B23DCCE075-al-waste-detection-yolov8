"""S2 CoreSet / k-Center-Greedy (Sener & Savarese, ICLR 2018,
arXiv:1708.00489, Algorithm 1)."""
import numpy as np


def kcenter_greedy(labeled_feats: np.ndarray, cand_feats: np.ndarray,
                   budget: int) -> list[int]:
    # running-min over labeled columns, avoids the full |C|x|L|xd diff
    cand = np.asarray(cand_feats, dtype=float)
    budget = min(budget, len(cand))                      # can't pick > |cand|
    min_d = np.full(len(cand), np.inf)
    if len(labeled_feats):
        for col in np.asarray(labeled_feats, dtype=float):
            min_d = np.minimum(min_d, np.linalg.norm(cand - col, axis=1))
    picks: list[int] = []
    for _ in range(budget):
        i = int(np.argmax(min_d))
        picks.append(i)
        d = np.linalg.norm(cand - cand[i], axis=1)
        min_d = np.minimum(min_d, d)
        min_d[i] = -1.0                                   # never re-pick
    return picks


def select(model, candidates, labeled, budget, cfg, rng):
    if budget <= 0:
        return []
    from .device import device
    c = cfg["al"]["coreset"]
    assert c.get("distance", "l2") == "l2", \
        f'coreset.distance: only "l2" implemented (Sener Alg.1), got {c["distance"]!r}'
    imgsz = cfg["al"]["imgsz"]
    bs = cfg["al"]["acq_batch"]
    lab = sorted(labeled)
    cand = sorted(set(candidates) - set(labeled))

    def embed(paths):
        if not paths:
            return np.empty((0, 1))
        kw = {} if c["embed_layer"] is None else {"embed": [c["embed_layer"]]}
        vecs = []
        # bound the source list (Ultralytics eagerly converts list sources)
        for start in range(0, len(paths), bs):
            chunk = paths[start:start + bs]
            # rect=False => square imgsz => embeddings batch-invariant
            vecs.extend(e.detach().cpu().numpy().astype(float)
                        for e in model.embed(
                            chunk, imgsz=imgsz, device=device(), rect=False,
                            batch=bs, stream=True, verbose=False, **kw))
        v = np.vstack(vecs)
        if c["l2_normalize"]:
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    lab_f, cand_f = embed(lab), embed(cand)
    idx = kcenter_greedy(lab_f, cand_f, budget)
    return sorted(cand[i] for i in idx)
