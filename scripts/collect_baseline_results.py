from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-mode baseline final metrics into one comparison table.")
    parser.add_argument("--run-dir", type=Path, default=FINALLY_ROOT / "baselines" / "strict_baselines_v1")
    parser.add_argument("--modes", nargs="+", default=["rgb", "depth", "rgbd"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    rows = []
    for mode in args.modes:
        final_metrics_path = args.run_dir / mode / "metrics" / "final_metrics.json"
        if not final_metrics_path.exists():
            raise FileNotFoundError(final_metrics_path)
        summary = json.loads(final_metrics_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        val = summary["final_val_metrics_from_best_checkpoint"]
        test = summary["test_metrics_from_best_checkpoint"]
        rows.append(
            {
                "mode": mode,
                "best_epoch": summary["best_epoch"],
                "best_val_iou": summary["best_val_iou"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_f1": val["f1"],
                "val_iou": val["iou"],
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
                "test_iou": test["iou"],
                "test_specificity": test["specificity"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_tp": test["tp"],
                "test_fp": test["fp"],
                "test_tn": test["tn"],
                "test_fn": test["fn"],
            }
        )

    out_csv = args.run_dir / "combined_test_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_json = args.run_dir / "combined_summary.json"
    out_json.write_text(json.dumps({"run_dir": str(args.run_dir), "rows": rows, "summaries": summaries}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    modes = [row["mode"] for row in rows]
    x = range(len(modes))
    axes[0].bar(x, [row["val_iou"] for row in rows], label="Val IoU", alpha=0.8)
    axes[0].bar(x, [row["test_iou"] for row in rows], label="Test IoU", alpha=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(modes)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("IoU Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [row["val_f1"] for row in rows], label="Val F1", alpha=0.8)
    axes[1].bar(x, [row["test_f1"] for row in rows], label="Test F1", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(modes)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("F1 Comparison")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.run_dir / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"run_dir": str(args.run_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
