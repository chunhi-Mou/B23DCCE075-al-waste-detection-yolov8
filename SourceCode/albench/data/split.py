"""Iterative multi-label stratified split (Sechidis, Tsoumakas & Vlahavas,
ECML-PKDD 2011): greedy, rarest-label first, deterministic given the seed."""
import random
from collections import Counter

SPLITS = ("train", "val", "test")


def stratified_split(items: list[tuple[str, set[int]]],
                     ratios: list[float], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)  # seeded tie-break order

    n = len(items)
    label_total = Counter()
    for _, cls in items:
        for c in cls:
            label_total[c] += 1

    desired = {s: {c: ratios[i] * label_total[c] for c in label_total}
               for i, s in enumerate(SPLITS)}
    cap = {s: ratios[i] * n for i, s in enumerate(SPLITS)}
    got_lbl = {s: Counter() for s in SPLITS}
    got_n = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}

    order = sorted(range(n),
                   key=lambda i: (min((label_total[c] for c in items[i][1]),
                                      default=10**9), i))
    for i in order:
        name, cls = items[i]
        if cls:
            c = min(cls, key=lambda x: label_total[x])
            best = max(SPLITS, key=lambda s: (
                desired[s][c] - got_lbl[s][c],
                cap[s] - got_n[s],
                -SPLITS.index(s)))
        else:
            best = max(SPLITS, key=lambda s: (cap[s] - got_n[s],
                                              -SPLITS.index(s)))
        out[best].append(name)
        got_n[best] += 1
        for cc in cls:
            got_lbl[best][cc] += 1
    return out
