from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "path_planning_checkpoint_selection_v1.json"


METRICS = [
    "planning_safety_utility",
    "success_rate_collision_free",
    "path_found_rate",
    "collision_rate_among_found",
    "mean_map_iou",
    "threshold",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    clean = [x for x in values if not math.isnan(x)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def std(values: list[float]) -> float:
    clean = [x for x in values if not math.isnan(x)]
    if len(clean) <= 1:
        return 0.0
    mu = mean(clean)
    return float(math.sqrt(sum((x - mu) ** 2 for x in clean) / (len(clean) - 1)))


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def build_model_meta(expanded_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(model["name"]): model for model in expanded_config["models"]}


def enrich_rows(rows: list[dict[str, str]], model_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        meta = model_meta.get(model, {})
        out: dict[str, Any] = {
            "model": model,
            "family": meta.get("family", model),
            "display_name": meta.get("display_name", meta.get("family", model)),
            "role": meta.get("role", ""),
            "seed": meta.get("seed", ""),
            "checkpoint_selection": meta.get("checkpoint_selection", "fixed"),
            "pair_id": meta.get("pair_id", model),
        }
        for key, value in row.items():
            out[key] = value
        enriched.append(out)
    return enriched


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        out: dict[str, Any] = {key: value for key, value in zip(keys, group_key)}
        first = group_rows[0]
        out["display_name"] = first.get("display_name", first.get("family", ""))
        out["members"] = len(group_rows)
        out["models"] = ";".join(str(row["model"]) for row in group_rows)
        out["seeds"] = ";".join(str(row.get("seed", "")) for row in group_rows if str(row.get("seed", "")) != "")
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in group_rows]
            clean = [x for x in values if not math.isnan(x)]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = std(values)
            out[f"{metric}_min"] = min(clean, default=math.nan)
            out[f"{metric}_max"] = max(clean, default=math.nan)
        output.append(out)
    return sorted(
        output,
        key=lambda row: (
            str(row.get("family", "")),
            str(row.get("checkpoint_selection", "")),
        ),
    )


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        tag = str(row.get("checkpoint_selection", ""))
        if tag not in {"best", "last"}:
            continue
        grouped[(str(row["family"]), str(row["pair_id"]))][tag] = row

    deltas: list[dict[str, Any]] = []
    for (family, pair_id), pair in grouped.items():
        if "best" not in pair or "last" not in pair:
            continue
        best = pair["best"]
        last = pair["last"]
        out: dict[str, Any] = {
            "family": family,
            "display_name": best.get("display_name", family),
            "pair_id": pair_id,
            "seed": best.get("seed", ""),
            "best_model": best["model"],
            "last_model": last["model"],
        }
        for metric in METRICS:
            out[f"{metric}_best"] = to_float(best.get(metric))
            out[f"{metric}_last"] = to_float(last.get(metric))
            out[f"{metric}_delta_last_minus_best"] = to_float(last.get(metric)) - to_float(best.get(metric))
        deltas.append(out)
    return sorted(deltas, key=lambda row: (row["family"], str(row["seed"])))


def aggregate_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    outputs: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        out: dict[str, Any] = {
            "family": family,
            "display_name": family_rows[0].get("display_name", family),
            "pairs": len(family_rows),
            "pair_ids": ";".join(str(row["pair_id"]) for row in family_rows),
        }
        for metric in METRICS:
            values = [to_float(row[f"{metric}_delta_last_minus_best"]) for row in family_rows]
            out[f"{metric}_delta_mean"] = mean(values)
            out[f"{metric}_delta_std"] = std(values)
            out[f"{metric}_delta_min"] = min([x for x in values if not math.isnan(x)], default=math.nan)
            out[f"{metric}_delta_max"] = max([x for x in values if not math.isnan(x)], default=math.nan)
        outputs.append(out)
    return sorted(outputs, key=lambda row: to_float(row["planning_safety_utility_delta_mean"]), reverse=True)


def plot_checkpoint_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    scoped = [row for row in rows if row.get("checkpoint_selection") in {"best", "last"}]
    families = []
    for row in scoped:
        family = str(row["family"])
        if family not in families:
            families.append(family)
    tags = ["best", "last"]
    x = np.arange(len(families))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    metrics = [
        ("planning_safety_utility", "Safety utility"),
        ("success_rate_collision_free", "Collision-free success"),
        ("mean_map_iou", "Mask IoU"),
    ]
    colors = {"best": "#2563eb", "last": "#f97316"}
    for ax, (metric, title) in zip(axes, metrics):
        for offset, tag in [(-width / 2, "best"), (width / 2, "last")]:
            values = []
            errors = []
            for family in families:
                row = next(item for item in scoped if item["family"] == family and item["checkpoint_selection"] == tag)
                values.append(to_float(row[f"{metric}_mean"]))
                errors.append(to_float(row[f"{metric}_std"]))
            ax.bar(x + offset, values, width, yerr=errors, capsize=4, label=tag, color=colors[tag], alpha=0.86)
        ax.set_title(title)
        ax.set_xticks(x, families, rotation=25, ha="right")
        ax.set_ylim(-0.15 if metric == "planning_safety_utility" else 0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_delta_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["display_name"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    metrics = [
        ("planning_safety_utility", "Safety delta"),
        ("success_rate_collision_free", "Success delta"),
        ("mean_map_iou", "Mask IoU delta"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        values = [to_float(row[f"{metric}_delta_mean"]) for row in rows]
        errors = [to_float(row[f"{metric}_delta_std"]) for row in rows]
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in values]
        ax.bar(x, values, yerr=errors, capsize=4, color=colors, alpha=0.86)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def metric_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | checkpoint | members | safety utility | success | path found | collision/found | mask IoU | threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("checkpoint_selection") not in {"best", "last"}:
            continue
        members = int(row["members"])
        def cell(metric: str) -> str:
            if members > 1:
                return f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}"
            return fmt(row[f"{metric}_mean"])
        lines.append(
            f"| `{row['display_name']}` | `{row['checkpoint_selection']}` | {members} | "
            f"{cell('planning_safety_utility')} | {cell('success_rate_collision_free')} | "
            f"{cell('path_found_rate')} | {cell('collision_rate_among_found')} | "
            f"{cell('mean_map_iou')} | {cell('threshold')} |"
        )
    return "\n".join(lines)


def delta_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | pairs | safety delta | success delta | path found delta | collision/found delta | mask IoU delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pairs = int(row["pairs"])
        def cell(metric: str) -> str:
            if pairs > 1:
                return f"{fmt(row[f'{metric}_delta_mean'])} ± {fmt(row[f'{metric}_delta_std'])}"
            return fmt(row[f"{metric}_delta_mean"])
        lines.append(
            f"| `{row['display_name']}` | {pairs} | {cell('planning_safety_utility')} | "
            f"{cell('success_rate_collision_free')} | {cell('path_found_rate')} | "
            f"{cell('collision_rate_among_found')} | {cell('mean_map_iou')} |"
        )
    return "\n".join(lines)


def write_report(report_path: Path, compact_config: dict[str, Any], family_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], run_dir: Path) -> None:
    best_winners = sorted(
        [row for row in family_rows if row.get("checkpoint_selection") == "best"],
        key=lambda row: to_float(row["planning_safety_utility_mean"]),
        reverse=True,
    )
    last_winners = sorted(
        [row for row in family_rows if row.get("checkpoint_selection") == "last"],
        key=lambda row: to_float(row["planning_safety_utility_mean"]),
        reverse=True,
    )
    delta_winners = sorted(delta_rows, key=lambda row: to_float(row["planning_safety_utility_delta_mean"]), reverse=True)
    best_top = best_winners[0] if best_winners else None
    last_top = last_winners[0] if last_winners else None
    delta_top = delta_winners[0] if delta_winners else None

    lines = [
        "# 第十二步：checkpoint selection 路径规划对照实验报告",
        "",
        f"生成日期：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 实验目的",
        "",
        "第十一步发现：第十步的 stable v2 recipe 虽然像素级更稳，但路径规划 safety utility 并不好。第十二步专门检查 checkpoint 选择是否是原因之一：同一训练 run 内，比较按 validation IoU 保存的 `best_model.pth` 与训练结束时的 `last_model.pth`。",
        "",
        "本步仍然不训练新模型，只读取已有 checkpoint；阈值仍然只在 validation split 上选择，再报告 test split。",
        "",
        "## 实验设置",
        "",
        f"- run name：`{compact_config['run_name']}`。",
        "- 比较 checkpoint：`best_model.pth` vs `last_model.pth`。",
        "- 评估 split：`val` 和 `test`。",
        "- 阈值扫描：`0.05` 到 `0.75`，步长 `0.05`。",
        "- 阈值选择：validation split。",
        "- 最终报告：test split。",
        "- 安全效用：`planning_safety_utility = 2 * collision_free_success - path_found_rate`。",
        "- 参与模型族：RGB baseline、RGB-D concat baseline、第四步单次 semantic clean、第八步 original semantic recipe、第九步 stable v1、第十步 stable v2。",
        "",
        "## best vs last 的 safety-selected 结果",
        "",
        metric_table(family_rows),
        "",
        "## last - best 差值",
        "",
        "正值表示 `last_model.pth` 比 `best_model.pth` 更好；负值表示按 validation IoU 保存的 best checkpoint 更好。",
        "",
        delta_table(delta_rows),
        "",
        "## 核心发现",
        "",
    ]
    if best_top:
        lines.append(f"- best checkpoint 中，最高 safety utility 是 `{best_top['display_name']}`：{fmt(best_top['planning_safety_utility_mean'])}。")
    if last_top:
        lines.append(f"- last checkpoint 中，最高 safety utility 是 `{last_top['display_name']}`：{fmt(last_top['planning_safety_utility_mean'])}。")
    if delta_top:
        lines.append(f"- last 相对 best 改善最大的模型族是 `{delta_top['display_name']}`，safety utility delta 为 {fmt(delta_top['planning_safety_utility_delta_mean'])}。")
    lines.extend(
        [
            "- `last_model.pth` 对 original semantic recipe 和 stable v2 有帮助，但提升后的 safety utility 仍然低于 RGB baseline 的 best checkpoint，也远低于第四步单次 `semantic_clean` 的 best checkpoint。",
            "- 因此，第十一步的负结果不能只归因于 validation IoU checkpoint selection；checkpoint selection 有影响，但还需要继续检查概率校准、路径安全导向阈值选择或 path-aware checkpoint selection。",
            "",
            "这一步要回答的不是“哪个模型最终最好”，而是“checkpoint selection 是否解释第十一步的负结果”。如果 last checkpoint 普遍提升 safety utility，就说明按 validation IoU 选 best checkpoint 可能不适合路径规划；如果 last checkpoint 没有提升，则要继续查概率校准或 path-aware selection。",
            "",
            "## 产物",
            "",
            "- `configs/path_planning_checkpoint_selection_v1.json`：第十二步紧凑配置。",
            "- `configs/path_planning_checkpoint_selection_v1_expanded.json`：自动展开后交给阈值扫描脚本的配置。",
            "- `scripts/prepare_path_planning_checkpoint_selection.py`：紧凑配置展开脚本。",
            "- `scripts/summarize_path_planning_checkpoint_selection.py`：第十二步聚合和报告生成脚本。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_model_safety_summary_enriched.csv`：逐模型 safety-selected test 指标。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_family_safety_summary.csv`：family/checkpoint 级 mean/std 聚合表。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_selection_delta.csv`：每个 pair 的 last-best 差值。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_selection_delta_aggregate.csv`：family 级 last-best 差值聚合。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_family_safety_metrics.png`：best/last 指标对比图。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/checkpoint_delta_metrics.png`：last-best 差值图。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize checkpoint-selection path planning experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expanded-config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_config = read_json(args.config)
    expanded_path = args.expanded_config or Path(compact_config["outputs"]["expanded_config"])
    expanded_config = read_json(expanded_path)
    run_dir = args.run_dir or (Path(compact_config["outputs"]["root"]) / str(compact_config["run_name"]))
    report_path = Path(compact_config["outputs"]["report"])
    model_meta = build_model_meta(expanded_config)

    safety_rows = enrich_rows(read_csv(run_dir / "selected_safety_threshold_test_summary.csv"), model_meta)
    success_rows = enrich_rows(read_csv(run_dir / "selected_threshold_test_summary.csv"), model_meta)
    family_safety = aggregate(safety_rows, ["family", "checkpoint_selection"])
    family_success = aggregate(success_rows, ["family", "checkpoint_selection"])
    deltas = paired_deltas(safety_rows)
    delta_summary = aggregate_deltas(deltas)

    write_csv(run_dir / "checkpoint_model_safety_summary_enriched.csv", safety_rows)
    write_csv(run_dir / "checkpoint_model_success_summary_enriched.csv", success_rows)
    write_csv(run_dir / "checkpoint_family_safety_summary.csv", family_safety)
    write_csv(run_dir / "checkpoint_family_success_summary.csv", family_success)
    write_csv(run_dir / "checkpoint_selection_delta.csv", deltas)
    write_csv(run_dir / "checkpoint_selection_delta_aggregate.csv", delta_summary)
    plot_checkpoint_bars(run_dir / "checkpoint_family_safety_metrics.png", family_safety)
    plot_delta_bars(run_dir / "checkpoint_delta_metrics.png", delta_summary)
    write_report(report_path, compact_config, family_safety, delta_summary, run_dir)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "run_dir": str(run_dir),
                "family_safety_summary": str(run_dir / "checkpoint_family_safety_summary.csv"),
                "delta_summary": str(run_dir / "checkpoint_selection_delta_aggregate.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
