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
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "path_planning_stability_v2_confirm.json"
DEFAULT_REPORT = FINALLY_ROOT / "reports" / "path_planning_stability_v2_confirm_summary.md"


METRICS = [
    "planning_safety_utility",
    "success_rate_collision_free",
    "path_found_rate",
    "collision_rate_among_found",
    "mean_map_iou",
    "threshold",
]


DISPLAY_NAMES = {
    "oracle": "oracle_gt",
    "rgb_baseline": "RGB baseline",
    "rgbd_concat_baseline": "RGB-D concat baseline",
    "semantic_clean_step4_single": "Step4 semantic clean",
    "semantic_original_recipe": "Original semantic recipe",
    "semantic_stable_v1_lr1e4_clip10": "Stable v1 lr1e-4 clip1.0",
    "semantic_stable_v2_lr2e4_clip10": "Stable v2 lr2e-4 clip1.0",
}


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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def enrich_rows(rows: list[dict[str, str]], model_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        meta = model_meta.get(model, {})
        out: dict[str, Any] = {
            "model": model,
            "family": meta.get("family", model),
            "display_name": DISPLAY_NAMES.get(str(meta.get("family", model)), str(meta.get("family", model))),
            "role": meta.get("role", ""),
            "seed": meta.get("seed", ""),
        }
        for key, value in row.items():
            out[key] = value
        enriched.append(out)
    return enriched


def aggregate_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        first = family_rows[0]
        out: dict[str, Any] = {
            "family": family,
            "display_name": DISPLAY_NAMES.get(family, family),
            "members": len(family_rows),
            "models": ";".join(str(row["model"]) for row in family_rows),
            "seeds": ";".join(str(row.get("seed", "")) for row in family_rows if str(row.get("seed", "")) != ""),
        }
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in family_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = std(values)
            out[f"{metric}_min"] = min([x for x in values if not math.isnan(x)], default=math.nan)
            out[f"{metric}_max"] = max([x for x in values if not math.isnan(x)], default=math.nan)
        summary_rows.append(out)

    return sorted(
        summary_rows,
        key=lambda row: (
            to_float(row["planning_safety_utility_mean"]),
            to_float(row["success_rate_collision_free_mean"]),
            to_float(row["mean_map_iou_mean"]),
        ),
        reverse=True,
    )


def plot_family_metrics(path: Path, family_rows: list[dict[str, Any]]) -> None:
    labels = [str(row["display_name"]) for row in family_rows if str(row["family"]) != "oracle"]
    rows = [row for row in family_rows if str(row["family"]) != "oracle"]
    x = np.arange(len(rows))
    metrics = [
        ("planning_safety_utility", "Safety utility"),
        ("success_rate_collision_free", "Collision-free success"),
        ("mean_map_iou", "Mask IoU"),
    ]
    colors = ["#2563eb", "#0f766e", "#f97316"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    for ax, (metric, title), color in zip(axes, metrics, colors):
        means = [to_float(row[f"{metric}_mean"]) for row in rows]
        errors = [to_float(row[f"{metric}_std"]) for row in rows]
        ax.bar(x, means, yerr=errors, capsize=4, color=color, alpha=0.86)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], include_std: bool = True) -> str:
    lines = [
        "| family | members | threshold | safety utility | success | path found | collision/found | mask IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if include_std and int(row["members"]) > 1:
            threshold = f"{fmt(row['threshold_mean'], 2)} ± {fmt(row['threshold_std'], 2)}"
            safety = f"{fmt(row['planning_safety_utility_mean'])} ± {fmt(row['planning_safety_utility_std'])}"
            success = f"{fmt(row['success_rate_collision_free_mean'])} ± {fmt(row['success_rate_collision_free_std'])}"
            path_found = f"{fmt(row['path_found_rate_mean'])} ± {fmt(row['path_found_rate_std'])}"
            collision = f"{fmt(row['collision_rate_among_found_mean'])} ± {fmt(row['collision_rate_among_found_std'])}"
            map_iou = f"{fmt(row['mean_map_iou_mean'])} ± {fmt(row['mean_map_iou_std'])}"
        else:
            threshold = fmt(row["threshold_mean"], 2)
            safety = fmt(row["planning_safety_utility_mean"])
            success = fmt(row["success_rate_collision_free_mean"])
            path_found = fmt(row["path_found_rate_mean"])
            collision = fmt(row["collision_rate_among_found_mean"])
            map_iou = fmt(row["mean_map_iou_mean"])
        lines.append(
            f"| `{row['display_name']}` | {row['members']} | {threshold} | {safety} | {success} | {path_found} | {collision} | {map_iou} |"
        )
    return "\n".join(lines)


def model_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | family | threshold | safety utility | success | path found | collision/found | mask IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['model']}` | `{row['display_name']}` | {fmt(row['threshold'], 2)} | "
            f"{fmt(row['planning_safety_utility'])} | {fmt(row['success_rate_collision_free'])} | "
            f"{fmt(row['path_found_rate'])} | {fmt(row['collision_rate_among_found'])} | {fmt(row['mean_map_iou'])} |"
        )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    family_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    success_family_rows: list[dict[str, Any]],
) -> None:
    v2 = next((row for row in family_rows if row["family"] == "semantic_stable_v2_lr2e4_clip10"), None)
    original = next((row for row in family_rows if row["family"] == "semantic_original_recipe"), None)
    stable_v1 = next((row for row in family_rows if row["family"] == "semantic_stable_v1_lr1e4_clip10"), None)
    rgb = next((row for row in family_rows if row["family"] == "rgb_baseline"), None)
    rgbd = next((row for row in family_rows if row["family"] == "rgbd_concat_baseline"), None)

    def gain(a: dict[str, Any] | None, b: dict[str, Any] | None, metric: str) -> str:
        if a is None or b is None:
            return ""
        return fmt(to_float(a[f"{metric}_mean"]) - to_float(b[f"{metric}_mean"]))

    lines = [
        "# 第十一步：stable v2 路径规划确认实验报告",
        "",
        f"生成日期：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 实验目的",
        "",
        "第十步锁定了 `lr=0.0002, gradient_clip_norm=1.0, deterministic=true` 作为当前最稳健的 semantic clean 训练 recipe。第十一步把这个 recipe 接回路径规划闭环，验证它是否不仅在像素级 Test IoU 上更稳定，也能在下游 A* 路径规划中保持安全效用。",
        "",
        "本步不训练新模型，只读取已有 checkpoint；阈值仍然只在 validation split 上选择，再报告 test split 指标。",
        "",
        "## 实验设置",
        "",
        f"- run name：`{config['run_name']}`。",
        "- 评估 split：`val` 和 `test`。",
        "- 阈值扫描：`0.05` 到 `0.75`，步长 `0.05`。",
        "- 阈值选择：validation split。",
        "- 最终报告：test split。",
        "- 路径规划器：8 邻域 A*。",
        "- 安全效用：`planning_safety_utility = 2 * collision_free_success - path_found_rate`。",
        "- 对同一模型和样本只做一次网络推理，再在同一张 probability map 上扫描所有阈值。",
        "",
        "参与比较的模型族：",
        "",
        "- RGB-only baseline。",
        "- RGB-D concat baseline。",
        "- 第四步单次 `semantic_clean` 主方法。",
        "- 第八步 original semantic recipe 的 3 个 seed。",
        "- 第九步 stable v1 recipe 的 3 个 seed。",
        "- 第十步 stable v2 `lr2e4_clip10` recipe 的 3 个 seed。",
        "",
        "## 安全效用阈值选择后的 recipe 级结果",
        "",
        markdown_table(family_rows),
        "",
        "## stable v2 的关键对比",
        "",
    ]
    if v2 is not None:
        lines.extend(
            [
                f"- stable v2 safety utility mean/std：{fmt(v2['planning_safety_utility_mean'])} ± {fmt(v2['planning_safety_utility_std'])}。",
                f"- stable v2 collision-free success mean/std：{fmt(v2['success_rate_collision_free_mean'])} ± {fmt(v2['success_rate_collision_free_std'])}。",
                f"- stable v2 mask IoU mean/std：{fmt(v2['mean_map_iou_mean'])} ± {fmt(v2['mean_map_iou_std'])}。",
            ]
        )
    if v2 is not None and rgb is not None:
        lines.append(f"- 相比 RGB-only baseline，stable v2 safety utility 差值为 {gain(v2, rgb, 'planning_safety_utility')}。")
    if v2 is not None and rgbd is not None:
        lines.append(f"- 相比 RGB-D concat baseline，stable v2 safety utility 差值为 {gain(v2, rgbd, 'planning_safety_utility')}。")
    if v2 is not None and original is not None:
        lines.append(f"- 相比 original semantic recipe，stable v2 safety utility 差值为 {gain(v2, original, 'planning_safety_utility')}，同时 mask IoU 差值为 {gain(v2, original, 'mean_map_iou')}。")
    if v2 is not None and stable_v1 is not None:
        lines.append(f"- 相比 stable v1，stable v2 safety utility 差值为 {gain(v2, stable_v1, 'planning_safety_utility')}。")

    lines.extend(
        [
            "",
            "从结果看，stable v2 的像素级 mask IoU 仍然较高，但路径规划 safety utility 没有保持第七步单次 `semantic_clean` 的优势。它的 path found rate 更高、collision/found 也更高，说明它更容易规划出路径，但这些路径更容易撞出 ground-truth road。这个负结果很关键：论文不能把第十步的像素级稳定性直接等同于下游规划安全性。",
            "",
            "## 逐模型结果",
            "",
            model_markdown_table(model_rows),
            "",
            "## 只最大化 success 的参照",
            "",
            "下面这张表使用 validation 上的 `success_rate_collision_free` 直接选阈值。它通常会偏向过低阈值，因此只作为参照，不作为论文主结论。",
            "",
            markdown_table(success_family_rows),
            "",
            "## 严谨结论",
            "",
            "这一步是下游确认实验。它比单次路径规划评估更严格，因为 semantic recipe 不再只看一个 checkpoint，而是按 recipe 的 3 个 seed 报告 mean/std。当前结果表明：stable v2 recipe 改善了像素级训练稳定性，但没有在 validation-driven safety utility 路径规划指标上同步带来优势。",
            "",
            "因此后续论文路线应更谨慎：可以保留第十步作为训练稳定性分析，但不能直接把 stable v2 作为最终下游规划主模型。下一步更应该研究 checkpoint selection、概率校准或 path-aware selection，解释为什么第七步单次 `semantic_clean` 在路径安全上明显更好。",
            "",
            "同时需要说明：第十步选择 stable v2 recipe 时已经比较过多个 recipe 的 test 指标，因此第十一步更适合作为锁定 recipe 后的下游一致性验证，而不是完全独立的最终泛化证明。",
            "",
            "## 产物",
            "",
            f"- `configs/{Path(str(config.get('config_name', 'path_planning_stability_v2_confirm.json'))).name}`：第十一步路径规划确认实验配置。",
            "- `scripts/summarize_path_planning_confirmatory.py`：第十一步结果聚合和 Markdown 报告生成脚本。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/threshold_rows.csv`：每个 split、模型、样本、阈值的完整路径规划结果。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/threshold_summary.csv`：每个 split、模型、阈值的聚合结果。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/selected_safety_threshold_test_summary.csv`：validation 安全效用选阈值后的 test 指标。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/model_safety_summary_enriched.csv`：加入 family/seed 元信息后的逐模型 test 指标。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/family_safety_summary.csv`：recipe/family 级 mean/std 聚合表。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/family_safety_metrics.png`：recipe/family 级安全效用、success 和 mask IoU 图。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize path planning stable v2 confirmatory experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    config["config_name"] = args.config.name
    run_dir = args.run_dir or (Path(config["outputs"]["root"]) / str(config["run_name"]))
    model_meta = {str(model["name"]): model for model in config["models"]}

    safety_rows = enrich_rows(read_csv(run_dir / "selected_safety_threshold_test_summary.csv"), model_meta)
    success_rows = enrich_rows(read_csv(run_dir / "selected_threshold_test_summary.csv"), model_meta)
    family_safety = aggregate_by_family(safety_rows)
    family_success = aggregate_by_family(success_rows)

    write_csv(run_dir / "model_safety_summary_enriched.csv", safety_rows)
    write_csv(run_dir / "model_success_selected_summary_enriched.csv", success_rows)
    write_csv(run_dir / "family_safety_summary.csv", family_safety)
    write_csv(run_dir / "family_success_selected_summary.csv", family_success)
    plot_family_metrics(run_dir / "family_safety_metrics.png", family_safety)
    write_report(args.report, config, run_dir, family_safety, safety_rows, family_success)

    payload = {
        "report": str(args.report),
        "run_dir": str(run_dir),
        "family_safety_summary": str(run_dir / "family_safety_summary.csv"),
        "model_safety_summary": str(run_dir / "model_safety_summary_enriched.csv"),
        "family_safety_metrics": str(run_dir / "family_safety_metrics.png"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
