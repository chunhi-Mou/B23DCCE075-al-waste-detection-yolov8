"""Statistical tests for the AL comparison (Demšar, JMLR 2006): paired-t,
Wilcoxon, Friedman + Holm, Cohen's dz + t-CI. Guarded for small n."""
import math

import numpy as np
from scipy import stats as ss


def _vals(diffs: dict):
    return np.array([diffs[k] for k in sorted(diffs)], dtype=float)


def paired_t(diffs: dict):
    v = _vals(diffs)
    if v.size < 2:
        return math.nan, math.nan
    r = ss.ttest_1samp(v, 0.0)
    return float(r.statistic), float(r.pvalue)


def wilcoxon_signed(diffs: dict):
    v = _vals(diffs)
    if v.size < 1 or np.allclose(v, 0.0):
        return math.nan, math.nan
    try:
        r = ss.wilcoxon(v)
    except ValueError:
        return math.nan, math.nan
    return float(r.statistic), float(r.pvalue)


def friedman(cols: dict):
    series = [np.asarray(v, dtype=float) for v in cols.values()]
    if len(series) < 3 or any(s.size < 2 for s in series):
        return math.nan, math.nan
    r = ss.friedmanchisquare(*series)
    return float(r.statistic), float(r.pvalue)


def holm(pvalues: dict, alpha: float) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (label, p) in enumerate(items):
        p_adj = max(prev, min(1.0, (m - rank) * p))
        out[label] = {"p": p, "p_adj": p_adj, "reject": p_adj <= alpha}
        prev = p_adj
    return out


def cohen_dz(diffs: dict) -> float:
    v = _vals(diffs)
    if v.size < 2:
        return math.nan
    sd = v.std(ddof=1)
    if sd == 0:
        return math.inf if v.mean() != 0 else 0.0
    return float(v.mean() / sd)


def ci_mean(diffs: dict, alpha: float):
    v = _vals(diffs)
    if v.size < 2:
        return math.nan, math.nan
    m, se = v.mean(), ss.sem(v)
    h = se * ss.t.ppf(1 - alpha / 2, v.size - 1)
    return float(m - h), float(m + h)


def caveat(n: int) -> str:
    return (f"Paired tests over n={n} seeds. With n<6 the Wilcoxon "
            f"signed-rank test cannot reach p<0.05 (min two-sided p at "
            f"n=5 is 0.0625; n=3 is 0.25); n=6 is the minimum that can. "
            f"Treat significance as low-powered, report effect size + CI "
            f"(Demšar 2006).")
