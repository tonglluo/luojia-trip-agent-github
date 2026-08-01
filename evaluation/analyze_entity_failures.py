"""Validate manual adjudication of failed entity fields."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_VERDICTS = {
    "model_error",
    "matcher_false_negative",
    "ambiguous_partial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default=(
            "evaluation/reports/"
            "intent_holdout_v1_run2_results.json"
        ),
    )
    parser.add_argument(
        "--adjudication",
        default=(
            "evaluation/reports/"
            "intent_holdout_v1_entity_adjudication.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "evaluation/reports/"
            "intent_holdout_v1_entity_analysis.json"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    result_path = PROJECT_ROOT / args.result
    adjudication_path = PROJECT_ROOT / args.adjudication
    report = load_json(result_path)
    adjudications = load_json(adjudication_path)

    failed_pairs = {
        (row["id"], field)
        for row in report["rows"]
        for field, passed in row["entity_checks"].items()
        if not passed
    }
    adjudicated_pairs = [
        (item["id"], item["field"])
        for item in adjudications
    ]
    duplicate_pairs = sorted(
        pair
        for pair, count in Counter(adjudicated_pairs).items()
        if count > 1
    )
    missing_pairs = sorted(failed_pairs - set(adjudicated_pairs))
    extra_pairs = sorted(set(adjudicated_pairs) - failed_pairs)
    invalid_verdicts = sorted({
        item.get("verdict")
        for item in adjudications
        if item.get("verdict") not in VALID_VERDICTS
    })
    if duplicate_pairs or missing_pairs or extra_pairs or invalid_verdicts:
        raise SystemExit(json.dumps({
            "duplicate_pairs": duplicate_pairs,
            "missing_pairs": missing_pairs,
            "extra_pairs": extra_pairs,
            "invalid_verdicts": invalid_verdicts,
        }, ensure_ascii=False, indent=2))

    verdict_counts = Counter(
        item["verdict"]
        for item in adjudications
    )
    entity_correct = sum(
        row["entity_correct"]
        for row in report["rows"]
    )
    entity_total = sum(
        row["entity_total"]
        for row in report["rows"]
    )
    confirmed_equivalent = verdict_counts["matcher_false_negative"]
    ambiguous = verdict_counts["ambiguous_partial"]

    output = {
        "source_result": args.result,
        "dataset_sha256": report["dataset_sha256"],
        "failed_field_count": len(failed_pairs),
        "adjudication_complete": True,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "strict_entity_accuracy": round(
            entity_correct / entity_total,
            4,
        ),
        "semantic_accuracy_lower_bound": round(
            (entity_correct + confirmed_equivalent) / entity_total,
            4,
        ),
        "semantic_accuracy_upper_bound": round(
            (
                entity_correct
                + confirmed_equivalent
                + ambiguous
            )
            / entity_total,
            4,
        ),
        "entity_correct": entity_correct,
        "entity_total": entity_total,
        "items": adjudications,
        "interpretation": (
            "Strict accuracy is the frozen deterministic score. "
            "The semantic range is a post-hoc diagnostic only and must "
            "not replace the official frozen-set metric."
        ),
    }
    destination = PROJECT_ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
