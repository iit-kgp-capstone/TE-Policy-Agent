"""Validate Week 1 Gold Evaluation Set JSONL files against gold_eval_schema.json.

Usage (from D:\AI_IITKGB\TE-Policy-Agent\data\evaluation):
    python validate_eval_set.py gold_eval_starter.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator


def validate_file(jsonl_path: Path, schema_path: Path) -> int:
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)
    seen_ids = set()
    decision_counts = Counter()
    area_counts = Counter()
    error_count = 0
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[Line {line_no}] Invalid JSON: {exc}")
                error_count += 1
                continue

            for err in validator.iter_errors(record):
                path = ".".join(str(p) for p in err.path) or "<root>"
                print(f"[Line {line_no} | {record.get('query_id', '?')}] {path}: {err.message}")
                error_count += 1

            qid = record.get("query_id")
            if qid in seen_ids:
                print(f"[Line {line_no}] Duplicate query_id: {qid}")
                error_count += 1
            seen_ids.add(qid)

            decision = record.get("gold_output", {}).get("decision_category")
            if decision:
                decision_counts[decision] += 1
            area = record.get("responsibility_area")
            if area:
                area_counts[area] += 1

    print(f"\nChecked {total} records | Errors: {error_count}")
    print("Decision distribution:", dict(decision_counts))
    print("Responsibility coverage:", dict(area_counts))
    return 1 if error_count else 0


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "gold_eval_starter.jsonl"
    schema_file = base_dir / "gold_eval_schema.json"
    sys.exit(validate_file(target, schema_file))
