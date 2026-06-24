"""Initial labeled set (Settles 2009): round-robin >=1 image per class then
uniform random fill."""
import random


def stratified_initial(items, frac, rng):
    n = max(1, round(len(items) * frac))
    by_class: dict[int, list[str]] = {}
    for key, classes in items:
        for c in classes:
            by_class.setdefault(c, []).append(key)

    selected: set[str] = set()
    for c in sorted(by_class):
        if len(selected) >= n:
            break
        pool = sorted(set(by_class[c]) - selected)
        if pool:
            selected.add(rng.choice(pool))
    rest = sorted(k for k, _ in items if k not in selected)
    rng.shuffle(rest)
    for k in rest:
        if len(selected) >= n:
            break
        selected.add(k)
    return sorted(selected)
