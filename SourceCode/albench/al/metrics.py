"""AL metrics. AUBC = trapezoidal integral of mAP50 over the labeled
fraction / span = budget-averaged mAP50 (primary endpoint)."""
import math

import numpy as np


def aubc(rounds: list) -> float:
    pts = sorted((r["frac"], r["test_mAP50"]) for r in rounds)
    if len(pts) < 2:
        return math.nan
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    span = x[-1] - x[0]
    if span <= 0:
        return math.nan
    return float(np.trapezoid(y, x) / span)


def _at_cap(rounds: list, cap_round: int):
    for r in rounds:
        if r["round"] == cap_round:
            return r
    return None


def map_at_cap(rounds: list, cap_round: int) -> float:
    r = _at_cap(rounds, cap_round)
    return float(r["test_mAP50"]) if r else math.nan


def per_class_ap_at_cap(rounds: list, cap_round: int):
    r = _at_cap(rounds, cap_round)
    return list(r["ap"]) if r else [math.nan] * 5


def per_seed_metric(results, strategy: str, metric_fn) -> dict:
    out = {}
    for sd in results.seeds:
        key = (strategy, sd)
        if key in results.by:
            out[sd] = metric_fn(results.by[key])
    return out


def paired_diffs(results, strat_a: str, strat_b: str, metric_fn) -> dict:
    a = per_seed_metric(results, strat_a, metric_fn)
    b = per_seed_metric(results, strat_b, metric_fn)
    return {sd: a[sd] - b[sd] for sd in sorted(set(a) & set(b))
            if not (math.isnan(a[sd]) or math.isnan(b[sd]))}


def seeds_complete(results, strategy: str, cap_round: int) -> list:
    out = []
    for sd in results.seeds:
        key = (strategy, sd)
        if key in results.by:
            rs = {r["round"] for r in results.by[key]}
            if set(range(cap_round + 1)) <= rs:
                out.append(sd)
    return out
