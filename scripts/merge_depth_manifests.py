from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge depth generation chunk manifests.")
    parser.add_argument("--reports-dir", type=Path, default=FINALLY_ROOT / "reports")
    parser.add_argument("--glob", default="depth_generation_full_all_*.csv")
    parser.add_argument("--output-csv", type=Path, default=FINALLY_ROOT / "reports" / "depth_generation_full_manifest.csv")
    parser.add_argument("--output-json", type=Path, default=FINALLY_ROOT / "reports" / "depth_generation_full_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = sorted(args.reports_dir.glob(args.glob))
    if not inputs:
        raise FileNotFoundError(f"No manifest files matched {args.reports_dir / args.glob}")

    rows: list[dict] = []
    fieldnames: list[str] | None = None
    for path in inputs:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row["source_manifest"] = str(path)
                rows.append(row)

    assert fieldnames is not None
    output_fields = fieldnames + ["source_manifest"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    sample_ids = [row["sample_id"] for row in rows]
    duplicate_ids = sorted([sample_id for sample_id, count in Counter(sample_ids).items() if count > 1])
    failed = [row for row in rows if row["status"] == "failed"]
    summary = {
        "input_manifest_files": [str(path) for path in inputs],
        "output_csv": str(args.output_csv),
        "num_rows": len(rows),
        "num_unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": duplicate_ids,
        "status_counts": dict(statuses),
        "failed_sample_ids": [row["sample_id"] for row in failed],
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
