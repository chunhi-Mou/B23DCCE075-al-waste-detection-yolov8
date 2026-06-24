"""Export frozen splits into a self-contained 3-folder YOLO dataset (images + labels + data.yaml)."""
import argparse, shutil
from pathlib import Path
import yaml
from albench.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--out", default="export", help="output directory")
    a = ap.parse_args()

    cfg  = load_config(Path(a.config))
    sdir = Path(cfg["splits_dir"])
    out  = Path(a.out)

    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

        paths = (sdir / f"{split}.txt").read_text().splitlines()
        paths = [p for p in paths if p.strip()]

        for img_path in paths:
            img = Path(img_path)
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            shutil.copy2(img, out / split / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, out / split / "labels" / lbl.name)

        print(f"{split}: {len(paths)} images")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    cfg["classes"]["nc"],
        "names": cfg["classes"]["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml))
    print(f"exported -> {out}/")


if __name__ == "__main__":
    main()
