"""Replay new deterministic routing rules over saved raw LLM predictions.

This does not measure the revised prompt. It isolates the effect of local
post-processing without making another paid model call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.intention_agent import IntentionAgent
from evaluation.common import load_jsonl, utc_now_iso, write_json
from evaluation.metrics import (
    entity_field_score,
    multilabel_classification,
    schedule_to_priority_map,
)
from evaluation.run_intent_eval import INTENT_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/reports/intent_results.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_postprocess_replay.json",
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Optional current JSONL answer set. When provided, expected "
            "intents, priorities and entities are read from this file."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(
        (PROJECT_ROOT / args.input).read_text(encoding="utf-8"),
    )
    dataset_path = PROJECT_ROOT / args.dataset if args.dataset else None
    cases_by_id = {}
    if dataset_path:
        cases_by_id = {
            case["id"]: case
            for case in load_jsonl(dataset_path)
        }
    replay_rows = []
    for row in source.get("rows", []):
        case = cases_by_id.get(row.get("id"), row)
        context = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')}"
            for item in case.get("history", [])
        )
        normalized = IntentionAgent._normalize_result(
            copy.deepcopy(row.get("raw_prediction", {})),
            case.get("query", row.get("query", "")),
            context,
        )
        predicted_intents = {
            item.get("type")
            for item in normalized.get("intents", [])
            if item.get("type")
        }
        predicted_schedule = schedule_to_priority_map(
            normalized.get("agent_schedule", []),
        )
        expected_intents = set(case.get("expected_intents", []))
        expected_schedule = case.get("expected_priorities", {})
        entity_correct, entity_total, entity_checks = entity_field_score(
            case.get("expected_entities", {}),
            normalized.get("key_entities", {}),
        )
        replay_rows.append({
            "id": row.get("id"),
            "expected_intents": sorted(expected_intents),
            "predicted_intents": sorted(predicted_intents),
            "intent_exact": expected_intents == predicted_intents,
            "expected_priorities": expected_schedule,
            "predicted_priorities": predicted_schedule,
            "schedule_exact": expected_schedule == predicted_schedule,
            "entity_correct": entity_correct,
            "entity_total": entity_total,
            "entity_checks": entity_checks,
        })

    classification = multilabel_classification(
        [set(row["expected_intents"]) for row in replay_rows],
        [set(row["predicted_intents"]) for row in replay_rows],
        INTENT_LABELS,
    )
    entity_correct = sum(row["entity_correct"] for row in replay_rows)
    entity_total = sum(row["entity_total"] for row in replay_rows)
    report = {
        "evaluation_type": "intent_postprocess_replay",
        "generated_at": utc_now_iso(),
        "source": args.input,
        "dataset": args.dataset or source.get("dataset"),
        "dataset_sha256": (
            hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            if dataset_path
            else source.get("dataset_sha256")
        ),
        "sample_count": len(replay_rows),
        "metrics": {
            **classification,
            "schedule_exact_match": round(
                sum(row["schedule_exact"] for row in replay_rows)
                / len(replay_rows),
                4,
            ) if replay_rows else 0.0,
            "entity_field_accuracy": round(
                entity_correct / entity_total if entity_total else 0.0,
                4,
            ),
        },
        "remaining_failures": [
            row
            for row in replay_rows
            if (
                not row["intent_exact"]
                or not row["schedule_exact"]
                or not all(row["entity_checks"].values())
            )
        ],
        "caveat": (
            "Offline replay of saved LLM outputs; it validates deterministic "
            "post-processing only and is not a revised-model evaluation."
        ),
    }
    destination = write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["remaining_failures"], ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
