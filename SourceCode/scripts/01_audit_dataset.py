"""Audit the configured dataset -> reports/dataset_audit.md."""
from pathlib import Path
from albench.config import load_config
from albench.data.labels import scan_dataset
from albench.data.audit import summarize, render_markdown


def main() -> None:
    cfg = load_config(Path("configs/benchmark.yaml"))
    root = Path(cfg["dataset"]["root"])
    ds = scan_dataset(root / cfg["dataset"]["images_dir"],
                      root / cfg["dataset"]["labels_dir"])
    summary = summarize(ds, cfg["classes"]["names"])
    out = Path(cfg["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.md").write_text(render_markdown(summary))
    print(f"n_paired={summary['n_paired']} "
          f"orphans(img/lbl)={summary['images_without_labels']}/"
          f"{summary['labels_without_images']} "
          f"box_counts={summary['box_counts']}")


if __name__ == "__main__":
    main()
