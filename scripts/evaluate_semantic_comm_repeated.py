from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train_baselines import dataloader_for, ensure_dir, load_json, read_ids, run_epoch
from train_semantic_comm import make_model


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"


def set_repeat_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path | None) -> list[dict]:
    if path is not None:
        return load_json(path)["models"]
    root = FINALLY_ROOT / "semantic_comm_runs"
    return [
        {
            "name": "clean",
            "variant": "clean",
            "run_dir": str(root / "semantic_comm_v1" / "clean"),
            "checkpoint": str(root / "semantic_comm_v1" / "clean" / "checkpoints" / "best_model.pth"),
            "source": "clean upper bound",
        },
        {
            "name": "quant8_ft",
            "variant": "quant8",
            "run_dir": str(root / "semantic_comm_v1_channel_finetune" / "quant8"),
            "checkpoint": str(root / "semantic_comm_v1_channel_finetune" / "quant8" / "checkpoints" / "best_model.pth"),
            "source": "clean initialized channel fine-tune",
        },
        {
            "name": "noise05_ft",
            "variant": "noise05",
            "run_dir": str(root / "semantic_comm_v1_channel_finetune" / "noise05"),
            "checkpoint": str(root / "semantic_comm_v1_channel_finetune" / "noise05" / "checkpoints" / "best_model.pth"),
            "source": "clean initialized channel fine-tune",
        },
        {
            "name": "dropout10_ft",
            "variant": "dropout10",
            "run_dir": str(root / "semantic_comm_v1_channel_finetune" / "dropout10"),
            "checkpoint": str(root / "semantic_comm_v1_channel_finetune" / "dropout10" / "checkpoints" / "best_model.pth"),
            "source": "clean initialized channel fine-tune",
        },
    ]


def variant_lookup(config: dict) -> dict[str, dict]:
    return {variant["name"]: variant for variant in config["variants"]}


def metric_row(model_name: str, split: str, repeat: int, seed: int, metrics) -> dict:
    return {
        "model": model_name,
        "split": split,
        "repeat": repeat,
        "seed": seed,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "iou": metrics.iou,
        "specificity": metrics.specificity,
        "balanced_accuracy": metrics.balanced_accuracy,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    metric_names = ["loss", "accuracy", "precision", "recall", "f1", "iou", "specificity", "balanced_accuracy"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["split"]), []).append(row)
    out = []
    for (model_name, split), group_rows in sorted(groups.items()):
        item = {
            "model": model_name,
            "split": split,
            "repeats": len(group_rows),
        }
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in group_rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
            item[f"{metric}_min"] = float(values.min())
            item[f"{metric}_max"] = float(values.max())
        out.append(item)
    return out


def plot_aggregate(path: Path, aggregate_rows: list[dict]) -> None:
    test_rows = [row for row in aggregate_rows if row["split"] == "test"]
    labels = [row["model"] for row in test_rows]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, [row["iou_mean"] for row in test_rows], yerr=[row["iou_std"] for row in test_rows], color="#2563eb", capsize=4)
    axes[0].set_title("Repeated Test IoU")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [row["f1_mean"] for row in test_rows], yerr=[row["f1_std"] for row in test_rows], color="#059669", capsize=4)
    axes[1].set_title("Repeated Test F1")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatedly evaluate semantic communication checkpoints under stochastic channels.")
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG_PATH)
    parser.add_argument("--semantic-config", type=Path, default=SEMANTIC_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=FINALLY_ROOT / "semantic_comm_runs" / "repeated_channel_eval_v1")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    args = parser.parse_args()

    protocol_config = load_json(args.protocol_config)
    semantic_config = load_json(args.semantic_config)
    models = load_manifest(args.manifest)
    variants = variant_lookup(semantic_config)
    output_dir = args.output_dir
    ensure_dir(output_dir)
    (output_dir / "evaluation_manifest.json").write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")
    (output_dir / "semantic_config_snapshot.json").write_text(json.dumps(semantic_config, indent=2), encoding="utf-8")
    (output_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    outputs_root = Path(protocol_config["outputs_root"])
    ids_by_split = {
        "val": read_ids(outputs_root / "protocol" / "val.txt"),
        "test": read_ids(outputs_root / "protocol" / "test.txt"),
    }
    batch_size = int(args.batch_size or semantic_config["batch_size"])
    loaders = {}
    for split in args.splits:
        _dataset, loader = dataloader_for(
            ids_by_split[split],
            "rgbd",
            split,
            protocol_config,
            semantic_config,
            batch_size,
            False,
            0,
        )
        loaders[split] = loader

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    threshold = float(semantic_config["threshold"])

    for model_spec in models:
        variant_name = model_spec["variant"]
        if variant_name not in variants:
            raise ValueError(f"Unknown variant in manifest: {variant_name}")
        variant = variants[variant_name]
        checkpoint_path = Path(model_spec["checkpoint"])
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = make_model(semantic_config, variant).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        metadata = ckpt.get("metadata", {})
        pos_weight = float(metadata.get("pos_weight", 1.0))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        for repeat in range(args.repeats):
            seed = int(args.seed + repeat)
            set_repeat_seed(seed)
            for split, loader in loaders.items():
                metrics = run_epoch(model, loader, device, criterion, threshold)
                rows.append(metric_row(model_spec["name"], split, repeat, seed, metrics))
                print(
                    f"{model_spec['name']} split={split} repeat={repeat + 1}/{args.repeats} "
                    f"iou={metrics.iou:.4f} f1={metrics.f1:.4f}"
                )

    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "per_repeat_metrics.csv", rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    plot_aggregate(output_dir / "repeated_channel_eval_plot.png", aggregate_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"per_repeat_rows": rows, "aggregate_rows": aggregate_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "aggregate_rows": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
