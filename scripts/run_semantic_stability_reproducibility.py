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
CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_stability_reproducibility.json"
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


def apply_overrides(base_config: dict[str, Any], seed: int, overrides: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["seed"] = int(seed)
    for key, value in overrides.items():
        config[key] = value
    return config


def semantic_completed(run_name: str, variants: list[str], semantic_root: Path) -> bool:
    return all((semantic_root / run_name / variant / "metrics" / "final_metrics.json").exists() for variant in variants)


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


def collect_semantic_rows(run_name: str, variants: list[str], semantic_root: Path, seed: int, experiment: str) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        path = semantic_root / run_name / variant / "metrics" / "final_metrics.json"
        payload = load_json(path)
        test = payload["test_metrics_from_best_checkpoint"]
        val = payload["final_val_metrics_from_best_checkpoint"]
        metadata = payload.get("metadata", {})
        rows.append(
            {
                "experiment": experiment,
                "model": "semantic_clean" if variant == "clean" else f"semantic_{variant}",
                "variant": variant,
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
                "learning_rate": metadata.get("learning_rate", ""),
                "deterministic": metadata.get("deterministic", ""),
                "gradient_clip_norm": metadata.get("gradient_clip_norm", ""),
                "scheduler_patience": metadata.get("scheduler_patience", ""),
                "metrics_path": str(path),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]], group_key: str = "experiment") -> list[dict[str, Any]]:
    groups = sorted({row[group_key] for row in rows})
    metrics = ["best_epoch", "best_val_iou", "val_iou", "val_f1", "test_iou", "test_f1", "test_precision", "test_recall", "test_accuracy", "elapsed_seconds"]
    output = []
    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        out: dict[str, Any] = {
            group_key: group,
            "runs": len(group_rows),
            "seeds": ";".join(str(row["seed"]) for row in group_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in {"", None}]
            out[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else 0.0
            out[f"{metric}_min"] = min(values) if values else math.nan
            out[f"{metric}_max"] = max(values) if values else math.nan
        output.append(out)
    return output


def read_reference_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "semantic_clean":
                rows.append(
                    {
                        "experiment": label,
                        "model": "semantic_clean",
                        "variant": row.get("mode_or_variant", "clean"),
                        "run_name": row.get("run_name", ""),
                        "seed": int(float(row["seed"])),
                        "best_epoch": float(row["best_epoch"]),
                        "best_val_iou": float(row["best_val_iou"]),
                        "val_iou": float(row["val_iou"]),
                        "val_f1": float(row["val_f1"]),
                        "test_iou": float(row["test_iou"]),
                        "test_f1": float(row["test_f1"]),
                        "test_precision": float(row["test_precision"]),
                        "test_recall": float(row["test_recall"]),
                        "test_accuracy": float(row["test_accuracy"]),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                        "metrics_path": row.get("metrics_path", ""),
                    }
                )
    return rows


def plot_comparison(path: Path, aggregate: list[dict[str, Any]]) -> None:
    labels = [row["experiment"] for row in aggregate]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in (
        (axes[0], "test_iou", "Semantic clean Test IoU"),
        (axes[1], "test_f1", "Semantic clean Test F1"),
    ):
        means = [float(row[f"{metric}_mean"]) for row in aggregate]
        stds = [float(row[f"{metric}_std"]) for row in aggregate]
        ax.bar(x, means, yerr=stds, capsize=5, color=["#64748b", "#16a34a"][: len(labels)])
        ax.set_title(title)
        ax.set_xticks(list(x), labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stabilized semantic clean multi-seed reproducibility experiment.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(Path(config["base_config"]))
    semantic_root = Path(base_config["outputs"]["root"])
    generated_config_dir = Path(config["outputs"]["generated_config_dir"])
    ensure_dir(generated_config_dir)
    report_prefix = Path(config["outputs"]["report_prefix"])
    ensure_dir(report_prefix.parent)
    variants = [str(x) for x in config["variants"]]
    device = args.device or str(config.get("device", ""))
    skip_existing = bool(config.get("skip_existing_completed_runs", True)) and not args.force
    command_rows = []
    metric_rows = []

    for seed in [int(x) for x in config["seeds"]]:
        run_name = str(config["run_name_template"]).format(seed=seed)
        seed_config = apply_overrides(base_config, seed, config["config_overrides"])
        seed_config_path = generated_config_dir / f"semantic_stability_seed_{seed}.json"
        write_json(seed_config_path, seed_config)
        if skip_existing and semantic_completed(run_name, variants, semantic_root):
            command_rows.append(
                {
                    "seed": seed,
                    "run_name": run_name,
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
                str(seed_config_path),
                "--run-name",
                run_name,
                "--variants",
                *variants,
            ]
            if device:
                command.extend(["--device", device])
            result = run_command(command, args.dry_run)
            result.update({"seed": seed, "run_name": run_name})
            command_rows.append(result)
            if int(result["returncode"]) != 0:
                raise SystemExit(f"Semantic stability seed {seed} failed with return code {result['returncode']}")
        if not args.dry_run:
            metric_rows.extend(collect_semantic_rows(run_name, variants, semantic_root, seed, "semantic_clean_stable_recipe"))

    reference = config.get("reference", {})
    reference_rows = read_reference_rows(Path(reference.get("rows_csv", "")), str(reference.get("label", "semantic_clean_reference")))
    comparison_rows = reference_rows + metric_rows
    aggregate = aggregate_rows(metric_rows)
    comparison_aggregate = aggregate_rows(comparison_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_commands.csv"), command_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_rows.csv"), metric_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_aggregate.csv"), aggregate)
    write_csv(report_prefix.with_name(report_prefix.name + "_comparison_rows.csv"), comparison_rows)
    write_csv(report_prefix.with_name(report_prefix.name + "_comparison_aggregate.csv"), comparison_aggregate)
    summary = {
        "config": config,
        "commands": command_rows,
        "rows": metric_rows,
        "aggregate": aggregate,
        "comparison_aggregate": comparison_aggregate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(report_prefix.with_name(report_prefix.name + "_summary.json"), summary)
    if comparison_aggregate:
        plot_comparison(report_prefix.with_name(report_prefix.name + "_comparison_metrics.png"), comparison_aggregate)
    print(json.dumps({"aggregate": aggregate, "comparison_aggregate": comparison_aggregate, "commands": command_rows}, indent=2))


if __name__ == "__main__":
    main()
