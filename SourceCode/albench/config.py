"""Load and validate the central benchmark config."""
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    ratios = cfg.get("split", {}).get("ratios")
    if not (isinstance(ratios, list) and len(ratios) == 3):
        raise ValueError(f"split.ratios must be a 3-list, got {ratios!r}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios must sum to 1.0, got {sum(ratios)}")
    return cfg


def load_seeds(path: Path) -> list[int]:
    return list(yaml.safe_load(Path(path).read_text())["seeds"])
