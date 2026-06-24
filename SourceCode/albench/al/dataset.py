"""Pool listing, per-image class set, and per-round data.yaml writing."""
from pathlib import Path
import yaml

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def pool_images(data_yaml: str) -> list[str]:
    d = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(d.get("path", str(Path(data_yaml).parent)))
    train_dir = base / d["train"]
    return sorted(str(p.resolve()) for p in train_dir.iterdir()
                  if p.suffix.lower() in _IMG_EXT)


def image_classes(img_path: str) -> frozenset[int]:
    lbl = Path(img_path.replace("/images/", "/labels/"))
    lbl = lbl.with_suffix(".txt")
    if not lbl.exists():
        return frozenset()
    ids = set()
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(int(float(line.split()[0])))
    return frozenset(ids)


def write_round_data_yaml(base_data_yaml: str, train_txt: str, out_yaml: str) -> None:
    d = yaml.safe_load(Path(base_data_yaml).read_text())
    base = Path(d.get("path", str(Path(base_data_yaml).parent)))
    d["train"] = str(Path(train_txt).resolve())
    d["val"] = str((base / d["val"]).resolve())
    d["test"] = str((base / d["test"]).resolve())
    d.pop("path", None)
    Path(out_yaml).write_text(yaml.safe_dump(d))
