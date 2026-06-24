"""S0 Random sampling: uniform i.i.d. draw from the unlabeled pool
(Settles 2009, AL Literature Survey, TR 1648)."""


def select(model, candidates, labeled, budget, cfg, rng):
    if budget <= 0:
        return []
    pool = sorted(set(candidates) - set(labeled))
    return sorted(rng.sample(pool, budget))
