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
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "constrained_path_aware_selection_v2.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value: Any, default: float = 0.0) -> float:
    out = to_float(value)
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def mean(values: list[Any]) -> float:
    clean = [to_float(x) for x in values]
    clean = [x for x in clean if not math.isnan(x)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def std(values: list[Any]) -> float:
    clean = [to_float(x) for x in values]
    clean = [x for x in clean if not math.isnan(x)]
    if len(clean) <= 1:
        return 0.0
    m = sum(clean) / len(clean)
    return float(math.sqrt(sum((x - m) ** 2 for x in clean) / (len(clean) - 1)))


def fmt(value: Any, digits: int = 4) -> str:
    out = to_float(value)
    if math.isnan(out):
        return "nan"
    return f"{out:.{digits}f}"


def spatial_risk(row: dict[str, Any], weights: dict[str, float], prefix: str = "") -> float:
    return (
        weights["collision_rate_weight"] * finite(row.get(f"{prefix}collision_rate_among_found"))
        + weights["corridor_fp_rate_weight"] * finite(row.get(f"{prefix}mean_path_corridor_fp_rate_found"))
        + weights["collision_bridge_area_weight"] * finite(row.get(f"{prefix}mean_collision_bridge_component_area_ratio_image_found"))
        + weights["blocked_rate_weight"] * finite(row.get(f"{prefix}start_or_goal_blocked_rate"))
    )


def pair_key(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("model"))


def make_selected_row(
    strategy: str,
    constraint_description: str,
    val_row: dict[str, str],
    test_row: dict[str, str],
    best_val_safety: float,
    eligible_count: int,
    total_threshold_count: int,
    weights: dict[str, float],
) -> dict[str, Any]:
    metrics = [
        "path_found_rate",
        "success_rate_collision_free",
        "unsafe_found_rate",
        "planning_safety_utility",
        "collision_rate_among_found",
        "start_or_goal_blocked_rate",
        "mean_map_iou",
        "mean_false_positive_rate",
        "mean_largest_fp_component_ratio_image",
        "mean_pred_start_component_fp_fraction",
        "mean_path_false_positive_fraction_found",
        "mean_path_corridor_fp_rate_found",
        "mean_collision_bridge_component_area_ratio_image_found",
        "path_aware_score",
    ]
    row: dict[str, Any] = {
        "strategy": strategy,
        "constraint_description": constraint_description,
        "model": test_row["model"],
        "family": test_row.get("family", ""),
        "display_name": test_row.get("display_name", ""),
        "checkpoint_selection": test_row.get("checkpoint_selection", ""),
        "pair_id": test_row.get("pair_id", ""),
        "seed": test_row.get("seed", ""),
        "selected_threshold": to_float(test_row["threshold"]),
        "best_val_safety_utility": best_val_safety,
        "val_safety_drop_from_best": best_val_safety - finite(val_row["planning_safety_utility"]),
        "eligible_threshold_count": eligible_count,
        "total_threshold_count": total_threshold_count,
        "val_spatial_risk_score": spatial_risk(val_row, weights),
        "test_spatial_risk_score": spatial_risk(test_row, weights),
    }
    for metric in metrics:
        row[f"val_{metric}"] = val_row.get(metric, math.nan)
        row[f"test_{metric}"] = test_row.get(metric, math.nan)
    return row


def select_constrained(
    summary_rows: list[dict[str, str]],
    config: dict[str, Any],
    strategy: dict[str, Any],
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_split = str(config["selection_split"])
    final_split = str(config["final_split"])
    abs_tol = float(strategy["absolute_safety_tolerance"])
    rel_floor = float(strategy["relative_safety_floor"])
    val_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_by_key: dict[tuple[str, str, float], dict[str, str]] = {}
    for row in summary_rows:
        threshold = to_float(row["threshold"])
        row_by_key[(row["split"], row["model"], threshold)] = row
        if row["split"] == selection_split:
            val_by_model[row["model"]].append(row)

    selected: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for model, val_rows in sorted(val_by_model.items()):
        best_val_safety = max(finite(row["planning_safety_utility"], -1e9) for row in val_rows)
        if best_val_safety > 0:
            floor = max(best_val_safety - abs_tol, best_val_safety * rel_floor)
        else:
            floor = best_val_safety - abs_tol
        candidates = [row for row in val_rows if finite(row["planning_safety_utility"], -1e9) >= floor - 1e-12]
        for row in val_rows:
            candidate_rows.append(
                {
                    "strategy": strategy["name"],
                    "model": model,
                    "family": row.get("family", ""),
                    "checkpoint_selection": row.get("checkpoint_selection", ""),
                    "threshold": row["threshold"],
                    "best_val_safety_utility": best_val_safety,
                    "safety_floor": floor,
                    "is_eligible": finite(row["planning_safety_utility"], -1e9) >= floor - 1e-12,
                    "val_planning_safety_utility": row["planning_safety_utility"],
                    "val_spatial_risk_score": spatial_risk(row, weights),
                    "val_collision_rate_among_found": row["collision_rate_among_found"],
                    "val_mean_path_corridor_fp_rate_found": row["mean_path_corridor_fp_rate_found"],
                    "val_mean_collision_bridge_component_area_ratio_image_found": row["mean_collision_bridge_component_area_ratio_image_found"],
                    "val_start_or_goal_blocked_rate": row["start_or_goal_blocked_rate"],
                }
            )

        def rank(row: dict[str, str]) -> tuple[float, float, float, float, float, float, float]:
            return (
                -spatial_risk(row, weights),
                finite(row["planning_safety_utility"], -1e9),
                finite(row["success_rate_collision_free"], -1e9),
                -finite(row["collision_rate_among_found"], 1.0),
                finite(row["mean_map_iou"], -1e9),
                -abs(finite(row["threshold"]) - 0.5),
                finite(row["threshold"]),
            )

        chosen_val = max(candidates, key=rank)
        threshold = to_float(chosen_val["threshold"])
        chosen_test = row_by_key[(final_split, model, threshold)]
        selected.append(
            make_selected_row(
                strategy["name"],
                str(strategy["description"]),
                chosen_val,
                chosen_test,
                best_val_safety,
                len(candidates),
                len(val_rows),
                weights,
            )
        )
    return selected, candidate_rows


def normalize_reference_rows(rows: list[dict[str, str]], strategy_names: set[str], weights: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["strategy"] not in strategy_names:
            continue
        item: dict[str, Any] = dict(row)
        item["constraint_description"] = "reference strategy from step 14"
        item["best_val_safety_utility"] = ""
        item["val_safety_drop_from_best"] = ""
        item["eligible_threshold_count"] = ""
        item["total_threshold_count"] = ""
        item["val_spatial_risk_score"] = spatial_risk(row, weights, prefix="val_")
        item["test_spatial_risk_score"] = spatial_risk(row, weights, prefix="test_")
        out.append(item)
    return out


def aggregate_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy"]), str(row["family"]), str(row.get("checkpoint_selection", "")))].append(row)
    metrics = [
        "test_planning_safety_utility",
        "test_success_rate_collision_free",
        "test_path_found_rate",
        "test_collision_rate_among_found",
        "test_mean_map_iou",
        "test_mean_path_corridor_fp_rate_found",
        "test_mean_collision_bridge_component_area_ratio_image_found",
        "test_spatial_risk_score",
    ]
    out: list[dict[str, Any]] = []
    for (strategy, family, checkpoint), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "strategy": strategy,
            "family": family,
            "checkpoint_selection": checkpoint,
            "members": len(group),
        }
        for metric in metrics:
            values = [row.get(metric, math.nan) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = std(values)
        out.append(item)
    return out


def pairwise_delta(rows: list[dict[str, Any]], baseline_strategy: str = "safety_utility") -> list[dict[str, Any]]:
    by_key = {(row["model"], row["strategy"]): row for row in rows}
    out: list[dict[str, Any]] = []
    strategies = sorted({row["strategy"] for row in rows if row["strategy"] != baseline_strategy})
    models = sorted({row["model"] for row in rows})
    for model in models:
        base = by_key.get((model, baseline_strategy))
        if base is None:
            continue
        for strategy in strategies:
            row = by_key.get((model, strategy))
            if row is None:
                continue
            out.append(
                {
                    "model": model,
                    "family": row.get("family", ""),
                    "checkpoint_selection": row.get("checkpoint_selection", ""),
                    "strategy": strategy,
                    "baseline_strategy": baseline_strategy,
                    "threshold_delta": finite(row["selected_threshold"]) - finite(base["selected_threshold"]),
                    "test_safety_delta": finite(row["test_planning_safety_utility"]) - finite(base["test_planning_safety_utility"]),
                    "test_collision_found_delta": finite(row["test_collision_rate_among_found"]) - finite(base["test_collision_rate_among_found"]),
                    "test_corridor_fp_delta": finite(row["test_mean_path_corridor_fp_rate_found"]) - finite(base["test_mean_path_corridor_fp_rate_found"]),
                    "test_spatial_risk_delta": finite(row["test_spatial_risk_score"]) - finite(base["test_spatial_risk_score"]),
                }
            )
    return out


def plot_strategy_bars(path: Path, family_rows: list[dict[str, Any]]) -> None:
    strategies = [
        "safety_utility",
        "constrained_safety_tie",
        "constrained_safety_2pp",
        "constrained_safety_5pp",
    ]
    labels = {
        "safety_utility": "safety",
        "constrained_safety_tie": "tie",
        "constrained_safety_2pp": "2pp",
        "constrained_safety_5pp": "5pp",
    }
    families = [
        "semantic_clean_step4_single",
        "rgb_baseline",
        "rgbd_concat_baseline",
        "semantic_original_recipe",
        "semantic_stable_v1_lr1e4_clip10",
        "semantic_stable_v2_lr2e4_clip10",
    ]
    rows = {
        (row["strategy"], row["family"], row["checkpoint_selection"]): row
        for row in family_rows
        if row["checkpoint_selection"] == "best"
    }
    x = np.arange(len(families))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    colors = ["#0f766e", "#2563eb", "#f97316", "#dc2626"]
    for idx, strategy in enumerate(strategies):
        safety = [finite(rows.get((strategy, family, "best"), {}).get("test_planning_safety_utility_mean"), math.nan) for family in families]
        risk = [finite(rows.get((strategy, family, "best"), {}).get("test_spatial_risk_score_mean"), math.nan) for family in families]
        axes[0].bar(x + (idx - 1.5) * width, safety, width=width, label=labels[strategy], color=colors[idx])
        axes[1].bar(x + (idx - 1.5) * width, risk, width=width, label=labels[strategy], color=colors[idx])
    axes[0].set_title("Test safety utility")
    axes[1].set_title("Test spatial risk score")
    for ax in axes:
        ax.set_xticks(x, families, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_sensitivity(path: Path, selected_rows: list[dict[str, Any]]) -> None:
    strategies = ["safety_utility", "constrained_safety_tie", "constrained_safety_2pp", "constrained_safety_5pp"]
    rows_by_strategy = {strategy: [row for row in selected_rows if row["strategy"] == strategy] for strategy in strategies}
    labels = ["safety", "tie", "2pp", "5pp"]
    safety_means = [mean([row["test_planning_safety_utility"] for row in rows_by_strategy[strategy]]) for strategy in strategies]
    risk_means = [mean([row["test_spatial_risk_score"] for row in rows_by_strategy[strategy]]) for strategy in strategies]
    threshold_means = [mean([row["selected_threshold"] for row in rows_by_strategy[strategy]]) for strategy in strategies]
    x = np.arange(len(strategies))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(x, safety_means, marker="o", color="#0f766e")
    axes[0].set_title("Mean test safety")
    axes[1].plot(x, risk_means, marker="o", color="#dc2626")
    axes[1].set_title("Mean spatial risk")
    axes[2].plot(x, threshold_means, marker="o", color="#2563eb")
    axes[2].set_title("Mean selected threshold")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    families = ["semantic_clean_step4_single", "rgb_baseline", "rgbd_concat_baseline", "semantic_original_recipe", "semantic_stable_v2_lr2e4_clip10"]
    strategies = ["safety_utility", "constrained_safety_tie", "constrained_safety_2pp", "constrained_safety_5pp"]
    by_key = {(row["strategy"], row["family"], row["checkpoint_selection"]): row for row in rows}
    lines = [
        "| strategy | family | ckpt | members | safety | collision/found | corridor FP | spatial risk |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        for strategy in strategies:
            row = by_key.get((strategy, family, "best"))
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{strategy}`",
                        f"`{family}`",
                        "`best`",
                        str(row["members"]),
                        fmt(row["test_planning_safety_utility_mean"]),
                        fmt(row["test_collision_rate_among_found_mean"]),
                        fmt(row["test_mean_path_corridor_fp_rate_found_mean"]),
                        fmt(row["test_spatial_risk_score_mean"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def summarize_deltas(delta_rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    rows = [row for row in delta_rows if row["strategy"] == strategy]
    improved_safety = [row for row in rows if finite(row["test_safety_delta"]) > 1e-12]
    reduced_collision = [row for row in rows if finite(row["test_collision_found_delta"]) < -1e-12]
    reduced_risk = [row for row in rows if finite(row["test_spatial_risk_delta"]) < -1e-12]
    return {
        "strategy": strategy,
        "models": len(rows),
        "mean_test_safety_delta": mean([row["test_safety_delta"] for row in rows]),
        "mean_collision_found_delta": mean([row["test_collision_found_delta"] for row in rows]),
        "mean_spatial_risk_delta": mean([row["test_spatial_risk_delta"] for row in rows]),
        "models_with_safety_improvement": len(improved_safety),
        "models_with_collision_reduction": len(reduced_collision),
        "models_with_spatial_risk_reduction": len(reduced_risk),
    }


def write_report(
    report_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    selected_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    delta_summary: list[dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    constrained = [row for row in selected_rows if row["strategy"].startswith("constrained")]
    best = max(constrained, key=lambda row: finite(row["test_planning_safety_utility"], -1e9))
    best_safety = max([row for row in selected_rows if row["strategy"] == "safety_utility"], key=lambda row: finite(row["test_planning_safety_utility"], -1e9))
    rel = run_dir.relative_to(FINALLY_ROOT).as_posix()
    lines = [
        "# 第十五步：Constrained path-aware selection v2 报告",
        "",
        "生成日期：2026-05-19",
        "",
        "## 实验目的",
        "",
        "第十四步发现直接用 path-aware score 选阈值没有超过 safety utility。第十五步把策略改得更谨慎：先要求 validation safety utility 接近该模型自己的最优值，再在候选阈值里选择空间风险最低的阈值。",
        "",
        "本步不重新训练模型，也不重复神经网络推理；它读取第十四步完整的模型-阈值空间诊断表，做后验阈值选择策略比较。",
        "",
        "## 策略定义",
        "",
        "- `safety_utility`：第十四步/第十二步同类策略，直接最大化 validation planning safety utility。",
        "- `constrained_safety_tie`：只允许 validation safety 最优阈值，用空间风险做 tie-breaker。",
        "- `constrained_safety_2pp`：允许 validation safety 最多下降 0.02，同时至少保留 95% 最优 safety，再最小化空间风险。",
        "- `constrained_safety_5pp`：允许 validation safety 最多下降 0.05，同时至少保留 90% 最优 safety，再最小化空间风险。",
        "",
        "空间风险分数：`0.45 * collision/found + 0.35 * path corridor FP + 0.10 * collision bridge area + 0.10 * blocked rate`。",
        "",
        "## family 级 best checkpoint 对比",
        "",
        markdown_table(family_rows),
        "",
        "## 核心发现",
        "",
        f"- constrained 策略中 test safety utility 最高的模型是 `{best['model']}`，策略 `{best['strategy']}`，阈值 {fmt(best['selected_threshold'])}，test safety utility {fmt(best['test_planning_safety_utility'])}。",
        f"- 纯 safety utility 的最高模型是 `{best_safety['model']}`，阈值 {fmt(best_safety['selected_threshold'])}，test safety utility {fmt(best_safety['test_planning_safety_utility'])}。",
    ]
    for item in delta_summary:
        lines.append(
            f"- `{item['strategy']}` 相对 `safety_utility`：mean safety delta {fmt(item['mean_test_safety_delta'])}，"
            f"mean collision/found delta {fmt(item['mean_collision_found_delta'])}，"
            f"mean spatial risk delta {fmt(item['mean_spatial_risk_delta'])}；"
            f"{item['models_with_collision_reduction']}/{item['models']} 个模型降低 collision/found。"
        )
    lines.extend(
        [
            "",
            "严谨结论：constrained selection v2 是更合理的阈值选择框架，但当前配置没有推翻第十四步结论。它适合写作方法诊断和消融：空间风险可以作为安全阈值选择的约束或 tie-breaker，但不能单独替代 safety utility。",
            "",
            "## 产物",
            "",
            "- `configs/constrained_path_aware_selection_v2.json`：第十五步 constrained selection 配置。",
            "- `scripts/summarize_constrained_path_aware_selection.py`：第十五步选择、汇总、绘图和报告脚本。",
            f"- `{rel}/config_snapshot.json`：配置快照。",
            f"- `{rel}/constrained_selected_thresholds.csv`：逐模型 constrained 策略选择结果。",
            f"- `{rel}/constrained_candidate_thresholds.csv`：每个模型、阈值是否满足 constrained 候选条件。",
            f"- `{rel}/selection_strategy_comparison_v2.csv`：参考策略与 constrained 策略合并对比。",
            f"- `{rel}/family_strategy_summary_v2.csv`：family/checkpoint 级 mean/std 聚合。",
            f"- `{rel}/strategy_delta_vs_safety.csv`：各策略相对 safety utility 的逐模型差值。",
            f"- `{rel}/strategy_delta_summary.csv`：各策略相对 safety utility 的整体差值摘要。",
            f"- `{rel}/constrained_strategy_comparison.png`：family 级 safety/risk 柱状图。",
            f"- `{rel}/constraint_sensitivity.png`：约束强度敏感性图。",
            f"- `{rel}/summary.json`：机器可读完整汇总。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize constrained path-aware threshold selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_name = str(config["run_name"])
    run_dir = Path(config["outputs"]["root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary_rows = read_csv(Path(config["source_spatial_summary_csv"]))
    reference_rows = read_csv(Path(config["source_strategy_comparison_csv"]))
    weights = {key: float(value) for key, value in config["spatial_risk_score"].items() if key.endswith("_weight")}

    constrained_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for strategy in config["constrained_strategies"]:
        selected, candidates = select_constrained(summary_rows, config, strategy, weights)
        constrained_rows.extend(selected)
        candidate_rows.extend(candidates)

    selected_reference_rows = normalize_reference_rows(reference_rows, set(config["reference_strategies"]), weights)
    all_strategy_rows = selected_reference_rows + constrained_rows
    family_rows = aggregate_strategy(all_strategy_rows)
    delta_rows = pairwise_delta(all_strategy_rows)
    delta_summary = [summarize_deltas(delta_rows, strategy["name"]) for strategy in config["constrained_strategies"]]

    write_csv(run_dir / "constrained_selected_thresholds.csv", constrained_rows)
    write_csv(run_dir / "constrained_candidate_thresholds.csv", candidate_rows)
    write_csv(run_dir / "selection_strategy_comparison_v2.csv", all_strategy_rows)
    write_csv(run_dir / "family_strategy_summary_v2.csv", family_rows)
    write_csv(run_dir / "strategy_delta_vs_safety.csv", delta_rows)
    write_csv(run_dir / "strategy_delta_summary.csv", delta_summary)
    plot_strategy_bars(run_dir / "constrained_strategy_comparison.png", family_rows)
    plot_sensitivity(run_dir / "constraint_sensitivity.png", all_strategy_rows)

    summary = {
        "run_name": run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_spatial_summary_csv": config["source_spatial_summary_csv"],
        "source_strategy_comparison_csv": config["source_strategy_comparison_csv"],
        "model_count": len({row["model"] for row in summary_rows}),
        "threshold_points": len(summary_rows),
        "constrained_selected_rows": len(constrained_rows),
        "delta_summary": delta_summary,
        "outputs": {
            "constrained_selected_thresholds": str(run_dir / "constrained_selected_thresholds.csv"),
            "constrained_candidate_thresholds": str(run_dir / "constrained_candidate_thresholds.csv"),
            "selection_strategy_comparison_v2": str(run_dir / "selection_strategy_comparison_v2.csv"),
            "family_strategy_summary_v2": str(run_dir / "family_strategy_summary_v2.csv"),
            "strategy_delta_vs_safety": str(run_dir / "strategy_delta_vs_safety.csv"),
            "strategy_delta_summary": str(run_dir / "strategy_delta_summary.csv"),
            "report": config["outputs"]["report"],
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(config["outputs"]["report"]), config, run_dir, all_strategy_rows, family_rows, delta_rows, delta_summary)
    print(f"Wrote constrained path-aware selection v2 summary to {run_dir}")


if __name__ == "__main__":
    main()
