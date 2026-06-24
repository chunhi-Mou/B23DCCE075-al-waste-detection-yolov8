"""Create frozen train/val/test manifests + YOLO data yaml + manifest hash."""
import argparse
from collections import Counter
from pathlib import Path
import yaml
from albench.config import load_config, load_seeds
from albench.data.labels import scan_dataset
from albench.data.split import stratified_split
from albench.repro import set_seed, sha256_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--seeds", default="configs/seeds.yaml")
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    seed = load_seeds(Path(a.seeds))[cfg["split"]["seed_index"]]
    set_seed(seed)

    root = Path(cfg["dataset"]["root"])
    ds = scan_dataset(root / cfg["dataset"]["images_dir"],
                      root / cfg["dataset"]["labels_dir"])
    items = [(str(it.image.resolve()), it.classes) for it in ds.items]
    parts = stratified_split(items, cfg["split"]["ratios"], seed)

    path2cls = {str(it.image.resolve()): it.classes for it in ds.items}
    dataset_classes = {c for cls_set in path2cls.values() for c in cls_set}
    floor = cfg["split"]["min_images_per_class_per_split"]
    for s, names in parts.items():
        cnt = Counter()
        for p in names:
            for c in path2cls[p]:
                cnt[c] += 1
        missing = [c for c in dataset_classes if cnt.get(c, 0) < floor]
        if missing:
            raise SystemExit(
                f"ACCEPTANCE FAIL: split '{s}' has classes {missing} "
                f"below floor {floor}: {dict(cnt)}")

    sdir = Path(cfg["splits_dir"])
    sdir.mkdir(parents=True, exist_ok=True)
    all_rel = []
    for s, names in parts.items():
        (sdir / f"{s}.txt").write_text("\n".join(names) + "\n")
        all_rel += names
    (sdir / "MANIFEST.sha256").write_text(
        f"seed={seed}\nsha256={sha256_manifest(all_rel)}\n")

    data_yaml = {
        "train": str((sdir / "train.txt").resolve()),
        "val": str((sdir / "val.txt").resolve()),
        "test": str((sdir / "test.txt").resolve()),
        "nc": cfg["classes"]["nc"],
        "names": cfg["classes"]["names"],
    }
    out_data_yaml = Path(a.config).parent / "data_baseline.yaml"
    out_data_yaml.write_text(yaml.safe_dump(data_yaml))
    print(f"seed={seed} sizes=" +
          " ".join(f"{s}:{len(v)}" for s, v in parts.items()))


if __name__ == "__main__":
    main()
