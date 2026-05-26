from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "path_planning_checkpoint_selection_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_file(base_dir: str, tag: str) -> Path:
    name = "best_model.pth" if tag == "best" else "last_model.pth"
    return Path(base_dir) / name


def clean_token(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def expand_model_group(group: dict[str, Any], checkpoint_tags: list[str]) -> list[dict[str, Any]]:
    members = group.get("members")
    if not members:
        members = [
            {
                "seed": group.get("seed", ""),
                "base_checkpoint_dir": group["base_checkpoint_dir"],
                "pair_id": group.get("pair_id", group["family"]),
            }
        ]

    rows: list[dict[str, Any]] = []
    for member in members:
        seed = member.get("seed", "")
        pair_id = member.get("pair_id", group.get("pair_id"))
        if not pair_id:
            pair_id = f"{group['family']}_seed{seed}" if seed != "" else str(group["family"])
        for tag in checkpoint_tags:
            checkpoint_path = checkpoint_file(str(member["base_checkpoint_dir"]), tag)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            name_parts = [str(group["family"])]
            if seed != "":
                name_parts.append(f"seed{seed}")
            name_parts.append(tag)
            model: dict[str, Any] = {
                "name": "_".join(name_parts),
                "type": group["type"],
                "family": group["family"],
                "display_name": group.get("display_name", group["family"]),
                "role": group.get("role", ""),
                "checkpoint_selection": tag,
                "pair_id": pair_id,
                "seed": seed,
                "checkpoint": checkpoint_path.as_posix(),
                "description": f"{group.get('display_name', group['family'])} {tag} checkpoint",
            }
            if group["type"] == "baseline":
                model["mode"] = group["mode"]
            elif group["type"] == "semantic":
                model["variant"] = group.get("variant", "clean")
            else:
                raise ValueError(f"Unsupported checkpoint group type: {group['type']}")
            rows.append(model)
    return rows


def build_expanded_config(config: dict[str, Any]) -> dict[str, Any]:
    expanded = {
        key: value
        for key, value in config.items()
        if key not in {"checkpoint_tags", "fixed_models", "checkpoint_groups"}
    }
    models: list[dict[str, Any]] = []
    models.extend(config.get("fixed_models", []))
    checkpoint_tags = [str(tag) for tag in config["checkpoint_tags"]]
    for group in config["checkpoint_groups"]:
        models.extend(expand_model_group(group, checkpoint_tags))
    expanded["models"] = models
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand checkpoint-selection path planning config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output or Path(config["outputs"]["expanded_config"])
    expanded = build_expanded_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(expanded, indent=2), encoding="utf-8")
    payload = {
        "source_config": str(args.config),
        "expanded_config": str(output),
        "models": len(expanded["models"]),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
