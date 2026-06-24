"""results.csv + confusion .npy -> figures, tables, RESULTS/SUMMARY/STATS/PROVENANCE."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albench.al import charts, health, metrics, ppal_charts, stats, tables
from albench.al.report_io import (load_confusion, load_results,
                                   schedule_fracs)
from albench.config import load_config

_PAIRS = [("uncertainty", "random"), ("coreset", "random"),
          ("ppal", "random"), ("uncertainty", "coreset"),
          ("ppal", "uncertainty"), ("ppal", "coreset")]


def _provenance(cfg, res, used_seeds) -> str:
    import numpy, scipy
    al = cfg["al"]
    return "\n".join([
        "AL benchmark — provenance", "",
        f"strategies: {res.strategies}",
        f"seeds present: {res.seeds}  (AUBC-complete: {used_seeds})",
        f"schedule: {al['schedule']}",
        f"numpy {numpy.__version__} scipy {scipy.__version__}",
        "Grounding: MI-AOD VOC arXiv:2104.02324; PPAL arXiv:2211.11612; "
        "Brust arXiv:1809.09875; Sener arXiv:1708.00489; Demšar JMLR 2006; "
        "Gashi arXiv:2403.14800.", ""])


def run(cfg: dict, csv_path: str, confusion_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alpha = cfg["al"]["analysis"]["alpha"]
    fracs = schedule_fracs(cfg["al"]["schedule"])
    R = len(fracs) - 1
    res = load_results(csv_path)

    # Coverage banner
    full = {s: metrics.seeds_complete(res, s, R) for s in res.strategies}
    is_partial = any(len(full[s]) < len(res.seeds) for s in res.strategies) \
        or not res.strategies
    cover_rows = [[c["strategy"], c["seed"], c["n_rounds"],
                   "yes" if c["seed"] in full.get(c["strategy"], []) else "no"]
                  for c in res.coverage]
    cover_md = tables.md_table(
        ["strategy", "seed", "rounds_done", "AUBC-complete"], cover_rows)

    # Metrics + stats
    aubc_ps = {s: metrics.per_seed_metric(res, s, metrics.aubc)
               for s in res.strategies}
    summary_rows, stat_lines = [], []
    for s in res.strategies:
        vals = [aubc_ps[s][sd] for sd in full[s]]
        mcap = [metrics.map_at_cap(res.by[(s, sd)], R) for sd in full[s]]
        summary_rows.append([
            charts.STRATEGY_STYLE.get(s, {"label": s})["label"],
            tables.fmt_pm(np.mean(vals) if vals else float("nan"),
                          np.std(vals) if vals else 0.0),
            tables.fmt_pm(np.mean(mcap) if mcap else float("nan"),
                          np.std(mcap) if mcap else 0.0),
            len(full[s])])
    pair_pt = {}
    for a, b in _PAIRS:
        if a not in res.strategies or b not in res.strategies:
            continue
        d = metrics.paired_diffs(res, a, b, metrics.aubc)
        t, pt = stats.paired_t(d)
        w, pw = stats.wilcoxon_signed(d)
        dz = stats.cohen_dz(d)
        lo, hi = stats.ci_mean(d, alpha)
        pair_pt[f"{a} vs {b}"] = pt
        stat_lines.append(
            f"{a} vs {b}: nΔ={len(d)}  meanΔ="
            f"{(sum(d.values())/len(d) if d else float('nan')):.4f}  "
            f"t={t:.3f} p_t={pt:.4f}  W={w:.1f} p_w={pw:.4f}  "
            f"dz={dz:.3f}  CI95=[{lo:.4f},{hi:.4f}]")
    # Holm family-wise correction
    valid_pt = {k: v for k, v in pair_pt.items()
                if not (v is None or math.isnan(v))}
    if valid_pt:
        hlm = stats.holm(valid_pt, alpha)
        stat_lines.append("Holm (paired-t; family = strategy pairs):")
        for k in sorted(hlm, key=lambda x: hlm[x]["p"]):
            stat_lines.append(
                f"  {k}: p={hlm[k]['p']:.4f} p_holm={hlm[k]['p_adj']:.4f} "
                f"reject={hlm[k]['reject']}")

    # Friedman + CD
    cd_made = None
    cols = {s: [aubc_ps[s][sd] for sd in sorted(set.intersection(
        *[set(full[x]) for x in res.strategies]))] for s in res.strategies} \
        if len(res.strategies) >= 3 and all(full.values()) else {}
    if cols and all(len(v) >= 2 for v in cols.values()):
        chi2, pf = stats.friedman(cols)
        stat_lines.append(f"Friedman: chi2={chi2:.3f} p={pf:.4f}")
        arr = np.array([cols[s] for s in res.strategies], dtype=float)
        ranks = (-arr).argsort(axis=0).argsort(axis=0) + 1   # best AUBC -> rank 1
        mean_ranks = {s: float(ranks[i].mean())
                      for i, s in enumerate(res.strategies)}
        cd_made = charts.cd_diagram(mean_ranks, len(next(iter(cols.values()))),
                                    str(out))

    # Figures
    charts.curve_mAP50(res, "test_mAP50", str(out), R)
    charts.curve_mAP50(res, "val_mAP50", str(out), R)
    charts.aubc_chart(res, str(out), R)
    charts.per_class_ap_chart(res, str(out), R)
    charts.class_balance_chart(res, str(out))
    for s in res.strategies:
        cm, used = load_confusion(confusion_dir, s, res.seeds)
        if cm is not None:
            charts.confusion_chart(cm, res.names, s, str(out))

    # PPAL adaptation diagnostics
    ppal_figs = []
    if ("ppal" in res.strategies
            and ppal_charts.has_quality(confusion_dir, res.seeds)):
        # each draw can return None on a partial run; link only written PNGs.
        ppal_figs = [p for p in (
            ppal_charts.ppal_difficulty_evo(confusion_dir, res.seeds,
                                            res.names, str(out), R),
            ppal_charts.ppal_difficulty_vs_ap(res, confusion_dir, res.seeds,
                                              res.names, str(out), R),
            ppal_charts.ppal_class_selection(res, str(out), R)) if p]
    # acq evidence written at a different loop point; gate separately
    if "ppal" in res.strategies and ppal_charts.has_acq(confusion_dir,
                                                        res.seeds):
        ppal_figs += [p for p in (
            ppal_charts.ppal_stage2_coverage(confusion_dir, res.seeds,
                                             str(out), R),
            ppal_charts.ppal_stage1_cutoff(confusion_dir, res.seeds,
                                           str(out), R)) if p]

    # Stability / no-overfit figures
    runs_al = cfg.get("al", {}).get("project", "runs/al")
    health.health_seed_variance(res, str(out))
    health.health_converge_heatmap(res, str(out))
    health.health_gap_overview(res, runs_al, str(out))
    base_glob = sorted(Path(cfg.get("baseline", {}).get(
        "project", "runs/baseline")).glob("baseline_frac1.0_*/results.csv"))
    if base_glob:
        health.health_baseline(base_glob[0], str(out))

    import csv as _csv
    aubc_tbl = [r[:3] for r in summary_rows]
    tex = tables.latex_table(["Strategy", "AUBC", "mAP@cap"], aubc_tbl,
                             caption="AUBC and mAP@cap (mean ± σ)",
                             label="tbl:aubc")
    (out / "tbl_aubc.tex").write_text(tex)
    with (out / "tbl_aubc.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["strategy", "AUBC", "mAP_at_cap", "n_seeds"])
        w.writerows(summary_rows)

    (out / "PROVENANCE").write_text(
        _provenance(cfg, res, {s: full[s] for s in res.strategies}))
    (out / "stats.txt").write_text(
        stats.caveat(len(res.seeds)) + "\n\n" + "\n".join(stat_lines) + "\n")
    (out / "coverage.txt").write_text(cover_md + "\n")
    print(f"report -> {out}/  partial={is_partial}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--data", default="export/data.yaml")
    ap.add_argument("--out", default="reports/al")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    rd = Path(cfg.get("reports_dir", "reports")) / "al"
    run(cfg, str(rd / "results.csv"), str(rd), a.out)


if __name__ == "__main__":
    main()
