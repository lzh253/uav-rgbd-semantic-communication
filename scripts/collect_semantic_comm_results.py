from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_summaries(run_dir: Path) -> list[dict]:
    summaries = []
    for final_metrics in sorted(run_dir.glob("*/metrics/final_metrics.json")):
        summaries.append(load_json(final_metrics))
    return sorted(summaries, key=lambda item: item["variant"])


def flatten_summary(summary: dict) -> dict:
    metrics = summary["test_metrics_from_best_checkpoint"]
    val_metrics = summary["final_val_metrics_from_best_checkpoint"]
    comm = summary["communication"]
    return {
        "variant": summary["variant"],
        "best_epoch": summary["best_epoch"],
        "best_val_iou": summary["best_val_iou"],
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_iou": metrics["iou"],
        "semantic_values": comm["semantic_values"],
        "value_compression_ratio": comm["value_compression_ratio"],
        "semantic_bits": comm["semantic_bits"],
        "bit_compression_ratio": comm["bit_compression_ratio"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict]) -> None:
    labels = [row["variant"] for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(x, [row["test_iou"] for row in rows], color="#2563eb")
    axes[0].set_title("Final Test IoU")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [row["test_f1"] for row in rows], color="#059669")
    axes[1].set_title("Final Test F1")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylim(0, 1)
    axes[2].bar(x, [row["bit_compression_ratio"] for row in rows], color="#7c3aed")
    axes[2].set_title("Bit Compression Ratio")
    axes[2].set_xticks(x, labels, rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect semantic communication final metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    summaries = collect_summaries(run_dir)
    rows = [flatten_summary(summary) for summary in summaries]
    write_csv(run_dir / "combined_test_metrics.csv", rows)
    (run_dir / "combined_summary.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")
    if rows:
        plot_comparison(run_dir / "semantic_comm_comparison.png", rows)
    print(json.dumps({"run_dir": str(run_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
