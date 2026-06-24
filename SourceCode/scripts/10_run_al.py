"""Run AL strategies across seeds."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from albench.config import load_config, load_seeds
from albench.al import loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--seeds", default="configs/seeds.yaml")
    ap.add_argument("--data", default="export/data.yaml")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--seed", default=None, type=int)
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    al = cfg["al"]
    strategies = [a.strategy] if a.strategy else al["strategies"]
    seed_list = ([a.seed] if a.seed is not None
                 else load_seeds(Path(a.seeds))[: al["seeds_n"]])

    failures = []
    for strategy in strategies:
        for seed in seed_list:
            print(f"{strategy} seed={seed}", flush=True)
            try:
                loop.run(strategy, seed, cfg, a.data)
            except Exception as e:
                print(f"FAILED: {strategy} seed={seed}: {e}", flush=True)
                failures.append((strategy, seed))
    if failures:
        for s, sd in failures:
            print(f"  failed: {s} seed={sd}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
