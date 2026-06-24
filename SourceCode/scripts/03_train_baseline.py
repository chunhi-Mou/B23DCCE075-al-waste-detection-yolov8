"""Train YOLOv8n baseline. --frac 1.0 = full train split; --frac 0.05 = headroom probe."""
import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from albench.config import load_config, load_seeds
from albench.repro import set_seed


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _val_map50_curve(results_csv: Path):
    if not results_csv.exists():
        return [], []
    rows = [{k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(results_csv.open())]
    if not rows:
        return [], []
    key = next((k for k in rows[0] if "mAP50" in k and "95" not in k), None)
    if not key:
        return [], []
    return ([int(r["epoch"]) for r in rows],
            [float(r[key]) for r in rows])


def _per_class_ap_lines(names, per_class: dict) -> str:
    out = []
    for i, n in enumerate(names):
        v = per_class.get(i)
        out.append(f"- **{n}**: {v:.4f}" if v is not None
                   else f"- **{n}**: n/a")
    return "\n".join(out)


def run(config, seeds, data, frac=1.0) -> None:
    cfg  = load_config(Path(config))
    seed = load_seeds(Path(seeds))[cfg["split"]["seed_index"]]
    set_seed(seed)

    data_yaml_path = Path(data)
    d = yaml.safe_load(data_yaml_path.read_text())
    base = Path(d.get("path", str(data_yaml_path.parent)))

    d["val"] = str((base / d["val"]).resolve())
    d["test"] = str((base / d["test"]).resolve())
    d.pop("path", None)

    train_imgs = sorted(
        str(p) for p in (base / d["train"]).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if frac < 1.0:
        k   = max(1, int(len(train_imgs) * frac))
        sub = random.Random(seed).sample(train_imgs, k)
        frac_dir = Path(cfg["splits_dir"])
        frac_dir.mkdir(parents=True, exist_ok=True)
        sub_txt = frac_dir / f"train_frac{frac}.txt"
        sub_txt.write_text("\n".join(sub) + "\n")
        d["train"] = str(sub_txt.resolve())
        data_yaml_path = frac_dir / f"data_frac{frac}.yaml"
        data_yaml_path.write_text(yaml.safe_dump(d))
    else:
        d["train"] = str((base / d["train"]).resolve())
        frac_dir = Path(cfg["splits_dir"])
        frac_dir.mkdir(parents=True, exist_ok=True)
        data_yaml_path = frac_dir / f"data_full_baseline.yaml"
        data_yaml_path.write_text(yaml.safe_dump(d))

    from ultralytics import YOLO
    b = cfg["baseline"]
    model = YOLO(b["model"])
    name = f"baseline_frac{frac}_seed{seed}"
    out = Path(b["project"]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    model.train(data=str(data_yaml_path), epochs=b["epochs"], imgsz=b["imgsz"],
                batch=b["batch"], cache=b.get("cache", False),
                device=_device(), project=str(out), name=name,
                pretrained=b.get("pretrained", True), patience=b.get("patience", 12),
                val=b.get("val", True),
                seed=seed, plots=True, exist_ok=True)

    run_dir = Path(model.trainer.save_dir)
    results_csv = run_dir / "results.csv"
    reports_out = Path(cfg["reports_dir"]) / "baseline" / name
    reports_out.mkdir(parents=True, exist_ok=True)

    metrics = model.val(data=str(data_yaml_path), split="test", device=_device())
    test_map50 = float(metrics.box.map50)
    per_class = {int(i): float(v)
                 for i, v in zip(metrics.box.ap_class_index,
                                 metrics.box.ap50)}
    names = list(d.get("names", []))

    ep, mp = _val_map50_curve(results_csv)
    best_map50 = 0.0
    best_epoch = 0
    if mp:
        best_i = max(range(len(mp)), key=lambda i: mp[i])
        best_map50 = mp[best_i]
        best_epoch = ep[best_i]

    if ep:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ep, mp, marker=".", label=f"frac={frac}")
        ax.set(xlabel="Epoch", ylabel="val mAP50",
               title=f"Baseline Convergence (frac={frac}, imgsz={b['imgsz']})")
        ax.legend()
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(reports_out / "convergence_curve.png", dpi=120)
        plt.close(fig)

    md = [
        f"# Baseline: {name}",
        "",
        f"- **Fraction**: {frac}",
        f"- **Images**: {len(train_imgs) if frac == 1.0 else k}",
        f"- **Model**: {b['model']}",
        f"- **Epochs (cap)**: {b['epochs']}",
        f"- **Patience**: {b.get('patience', 12)}",
        f"- **Best Val mAP50**: {best_map50:.4f} (@epoch {best_epoch})",
        f"- **Stopped @**: {ep[-1] if ep else 'N/A'}",
        f"- **Test mAP50**: {test_map50:.4f}",
        "",
        "## Full-data baseline" if frac == 1.0 else "## Subsampled baseline",
        f"Trained on {frac*100:.1f}% of the train split.",
        "",
        "## Per-class AP@50 (test)",
        _per_class_ap_lines(names, per_class),
    ]
    (reports_out / "SUMMARY.md").write_text("\n".join(md) + "\n")

    from albench.al.health import health_baseline
    health_baseline(results_csv, str(reports_out))

    print(f"\n{name} summary:")
    print(f"  Best val mAP50: {best_map50:.4f} @epoch {best_epoch}")
    print(f"  Test mAP50:     {test_map50:.4f}")
    print(f"  Weights/plots:  {run_dir}")
    print(f"  Report:         {reports_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--seeds",  default="configs/seeds.yaml")
    ap.add_argument("--data",   default="export/data.yaml",
                    help="dataset yaml (train/val/test folders)")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of train split (1.0=full, 0.05=probe)")
    a = ap.parse_args()
    run(a.config, a.seeds, a.data, a.frac)


if __name__ == "__main__":
    main()
