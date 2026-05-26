from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = FINALLY_ROOT / "configs" / "multi_seed_reproducibility.json"
BASELINE_SCRIPT = SCRIPT_DIR / "train_baselines.py"
SEMANTIC_SCRIPT = SCRIPT_DIR / "train_semantic_comm.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_seed_config(base_config_path: Path, seed: int, num_epochs: int, patience: int, min_delta: float) -> dict[str, Any]:
    config = load_json(base_config_path)
    config["seed"] = int(seed)
    config["num_epochs"] = int(num_epochs)
    config["early_stopping_patience"] = int(patience)
    config["early_stopping_min_delta"] = float(min_delta)
    return config


def run_command(command: list[str], dry_run: bool) -> dict[str, Any]:
    started = time.time()
    if dry_run:
        return {
            "command": " ".join(command),
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
        }
    completed = subprocess.run(command, cwd=str(FINALLY_ROOT))
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "elapsed_seconds": time.time() - started,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "dry_run": False,
    }


def baseline_completed(run_name: str, modes: list[str], baseline_root: Path) -> bool:
    return all((baseline_root / run_name / mode / "metrics" / "final_metrics.json").exists() for mode in modes)


def semantic_completed(run_name: str, variants: list[str], semantic_root: Path) -> bool:
    return all((semantic_root / run_name / variant / "metrics" / "final_metrics.json").exists() for variant in variants)


def collect_baseline_rows(run_name: str, modes: list[str], baseline_root: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for mode in modes:
        path = baseline_root / run_name / mode / "metrics" / "final_metrics.json"
        payload = load_json(path)
        test = payload["test_metrics_from_best_checkpoint"]
        val = payload["final_val_metrics_from_best_checkpoint"]
        rows.append(
            {
                "family": "baseline",
                "model": f"{mode}_baseline" if mode != "rgbd" else "rgbd_concat_baseline",
                "mode_or_variant": mode,
                "run_name": run_name,
                "seed": seed,
                "best_epoch": payload["best_epoch"],
                "best_val_iou": payload["best_val_iou"],
                "val_iou": val["iou"],
                "val_f1": val["f1"],
                "test_iou": test["iou"],
                "test_f1": test["f1"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_accuracy": test["accuracy"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "metrics_path": str(path),
            }
        )
    return rows


def collect_semantic_rows(run_name: str, variants: list[str], semantic_root: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        path = semantic_root / run_name / variant / "metrics" / "final_metrics.json"
        payload = load_json(path)
        test = payload["test_metrics_from_best_checkpoint"]
        val = payload["final_val_metrics_from_best_checkpoint"]
        communication = payload.get("communication", {})
        rows.append(
            {
                "family": "semantic",
                "model": "semantic_clean" if variant == "clean" else f"semantic_{variant}",
                "mode_or_variant": variant,
                "run_name": run_name,
                "seed": seed,
                "best_epoch": payload["best_epoch"],
                "best_val_iou": payload["best_val_iou"],
                "val_iou": val["iou"],
                "val_f1": val["f1"],
                "test_iou": test["iou"],
                "test_f1": test["f1"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_accuracy": test["accuracy"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "value_compression_ratio": communication.get("value_compression_ratio", ""),
                "bit_compression_ratio": communication.get("bit_compression_ratio", ""),
                "metrics_path": str(path),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({row["model"] for row in rows})
    aggregate = []
    metrics = ["best_epoch", "best_val_iou", "val_iou", "val_f1", "test_iou", "test_f1", "test_precision", "test_recall", "test_accuracy", "elapsed_seconds"]
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        out: dict[str, Any] = {
            "model": model,
            "runs": len(model_rows),
            "seeds": ";".join(str(row["seed"]) for row in model_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in model_rows if row.get(metric) not in {"", None}]
            out[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else 0.0
            out[f"{metric}_min"] = min(values) if values else math.nan
            out[f"{metric}_max"] = max(values) if values else math.nan
        aggregate.append(out)
    return aggregate


def plot_aggregate(path: Path, aggregate_rows_: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in aggregate_rows_]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in (
        (axes[0], "test_iou", "Test IoU across seeds"),
        (axes[1], "test_f1", "Test F1 across seeds"),
    ):
        means = [float(row[f"{metric}_mean"]) for row in aggregate_rows_]
        stds = [float(row[f"{metric}_std"]) for row in aggregate_rows_]
        ax.bar(x, means, yerr=stds, capsize=5, color=["#2563eb", "#0f766e", "#16a34a"][: len(labels)])
        ax.set_title(title)
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed reproducibility training for key models.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="")
    parser.add_argument("--force", action="store_true", help="Rerun training even when final metrics already exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    generated_config_dir = Path(config["outputs"]["generated_config_dir"])
    ensure_dir(generated_config_dir)
    report_prefix = Path(config["outputs"]["report_prefix"])
    ensure_dir(report_prefix.parent)
    device = args.device or str(config.get("device", ""))
    skip_existing = bool(config.get("skip_existing_completed_runs", True)) and not args.force
    command_rows: list[dict[str, Any]] = []

    baseline_cfg = config["baseline"]
    baseline_base_path = Path(baseline_cfg["base_config"])
    baseline_base = load_json(baseline_base_path)
    baseline_root = Path(baseline_base["outputs"]["root"])
    baseline_modes = [str(x) for x in baseline_cfg["modes"]]

    semantic_cfg = config["semantic"]
    semantic_base_path = Path(semantic_cfg["base_config"])
    semantic_base = load_json(semantic_base_path)
    semantic_root = Path(semantic_base["outputs"]["root"])
    semantic_variants = [str(x) for x in semantic_cfg["variants"]]

    for seed in [int(x) for x in config["seeds"]]:
        baseline_run_name = str(baseline_cfg["run_name_template"]).format(seed=seed)
        baseline_seed_config = make_seed_config(
            baseline_base_path,
            seed,
            int(baseline_cfg["num_epochs"]),
            int(baseline_cfg["early_stopping_patience"]),
            float(baseline_cfg["early_stopping_min_delta"]),
        )
        baseline_seed_config_path = generated_config_dir / f"baseline_seed_{seed}.json"
        write_json(baseline_seed_config_path, baseline_seed_config)
        if skip_existing and baseline_completed(baseline_run_name, baseline_modes, baseline_root):
            command_rows.append(
                {
                    "family": "baseline",
                    "seed": seed,
                    "run_name": baseline_run_name,
                    "command": "skipped_existing_completed_run",
                    "returncode": 0,
                    "elapsed_seconds": 0.0,
                }
            )
        else:
            command = [
                sys.executable,
                "-B",
                str(BASELINE_SCRIPT),
                "--baseline-config",
                str(baseline_seed_config_path),
                "--run-name",
                baseline_run_name,
                "--modes",
                *baseline_modes,
            ]
            if device:
                command.extend(["--device", device])
            result = run_command(command, args.dry_run)
            result.update({"family": "baseline", "seed": seed, "run_name": baseline_run_name})
            command_rows.append(result)
            if int(result["returncode"]) != 0:
                raise SystemExit(f"Baseline seed {seed} failed with return code {result['returncode']}")

        semantic_run_name = str(semantic_cfg["run_name_template"]).format(seed=seed)
        semantic_seed_config = make_seed_config(
            semantic_base_path,
            seed,
            int(semantic_cfg["num_epochs"]),
            int(semantic_cfg["early_stopping_patience"]),
            float(semantic_cfg["early_stopping_min_delta"]),
        )
        semantic_seed_config_path = generated_config_dir / f"semantic_seed_{seed}.json"
        write_json(semantic_seed_config_path, semantic_seed_config)
        if skip_existing and semantic_completed(semantic_run_name, semantic_variants, semantic_root):
            command_rows.append(
                {
                    "family": "semantic",
                    "seed": seed,
                    "run_name": semantic_run_name,
                    "command": "skipped_existing_completed_run",
                    "returncode": 0,
                    "elapsed_seconds": 0.0,
                }
            )
        else:
            command = [
                sys.executable,
                "-B",
                str(SEMANTIC_SCRIPT),
                "--semantic-config",
                str(semantic_seed_config_path),
                "--run-name",
                semantic_run_name,
                "--variants",
                *semantic_variants,
            ]
            if device:
                command.extend(["--device", device])
            result = run_command(command, args.dry_run)
            result.update({"family": "semantic", "seed": seed, "run_name": semantic_run_name})
            command_rows.append(result)
            if int(result["returncode"]) != 0:
                raise SystemExit(f"Semantic seed {seed} failed with return code {result['returncode']}")

    metric_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        for seed in [int(x) for x in config["seeds"]]:
            baseline_run_name = str(baseline_cfg["run_name_template"]).format(seed=seed)
            metric_rows.extend(collect_baseline_rows(baseline_run_name, baseline_modes, baseline_root, seed))
            semantic_run_name = str(semantic_cfg["run_name_template"]).format(seed=seed)
            metric_rows.extend(collect_semantic_rows(semantic_run_name, semantic_variants, semantic_root, seed))
    aggregate = aggregate_rows(metric_rows) if metric_rows else []
    write_csv(report_prefix.with_name(report_prefix.name + "_commands.csv"), command_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_rows.csv"), metric_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_aggregate.csv"), aggregate)
    summary = {
        "config": config,
        "commands": command_rows,
        "rows": metric_rows,
        "aggregate": aggregate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(report_prefix.with_name(report_prefix.name + "_summary.json"), summary)
    if aggregate:
        plot_aggregate(report_prefix.with_name(report_prefix.name + "_metrics.png"), aggregate)
    print(json.dumps({"aggregate": aggregate, "commands": command_rows}, indent=2))


if __name__ == "__main__":
    main()
