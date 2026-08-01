"""Evaluate multi-label intent recognition and agent scheduling."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.message import Msg

from agents.intention_agent import IntentionAgent
from config import LLM_CONFIG
from config_agentscope import init_agentscope
from evaluation.common import (
    build_chat_model,
    load_jsonl,
    utc_now_iso,
    write_json,
)
from evaluation.metrics import (
    entity_field_score,
    multilabel_classification,
    schedule_to_priority_map,
)


INTENT_LABELS = {
    "itinerary_planning",
    "memory_query",
    "preference",
    "rag_knowledge",
    "information_query",
    "event_collection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/intent_eval.sample.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_results.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
    )
    return parser.parse_args()


def is_retryable_rate_limit(exc: Exception) -> bool:
    message = str(exc).casefold()
    return (
        "429" in message
        or "rate limit" in message
        or "速率限制" in message
        or "'code': '1302'" in message
    )


def build_messages(case: dict) -> list[Msg]:
    messages: list[Msg] = []
    for index, item in enumerate(case.get("history", [])):
        role = item["role"]
        messages.append(
            Msg(
                name=f"history_{index}_{role}",
                role=role,
                content=item["content"],
            ),
        )
    messages.append(
        Msg(name="user", role="user", content=case["query"]),
    )
    return messages


async def run() -> int:
    args = parse_args()
    cases = load_jsonl(PROJECT_ROOT / args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    init_agentscope()
    model = build_chat_model(
        thinking_type=args.thinking,
        max_tokens=2500,
        temperature=0.1,
    )
    agent = IntentionAgent(name="IntentEvalAgent", model=model)

    rows = []
    for index, case in enumerate(cases, 1):
        started_at = time.perf_counter()
        error = None
        prediction: dict = {}
        attempt_count = 0
        rate_limit_retry_count = 0
        while attempt_count <= args.max_retries:
            attempt_count += 1
            try:
                response = await agent.reply(build_messages(case))
                prediction = json.loads(response.content)
                error = None
                break
            except Exception as exc:
                error = str(exc)
                if (
                    not is_retryable_rate_limit(exc)
                    or attempt_count > args.max_retries
                ):
                    break
                rate_limit_retry_count += 1
                retry_delay = (
                    args.retry_base_delay
                    * (2 ** (rate_limit_retry_count - 1))
                )
                print(
                    f"[{index}/{len(cases)}] {case['id']} "
                    f"rate-limited; retry "
                    f"{rate_limit_retry_count}/{args.max_retries} "
                    f"in {retry_delay:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)

        elapsed = time.perf_counter() - started_at
        predicted_intents = {
            item.get("type")
            for item in prediction.get("intents", [])
            if item.get("type")
        }
        predicted_schedule = schedule_to_priority_map(
            prediction.get("agent_schedule", []),
        )
        scheduled_agents = set(predicted_schedule)
        expected_schedule = {
            str(name): int(priority)
            for name, priority in case.get(
                "expected_priorities",
                {},
            ).items()
        }
        entity_correct, entity_total, entity_checks = entity_field_score(
            case.get("expected_entities", {}),
            prediction.get("key_entities", {}),
        )

        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "query": case["query"],
            "expected_intents": case["expected_intents"],
            "predicted_intents": sorted(predicted_intents),
            "intent_exact": (
                set(case["expected_intents"]) == predicted_intents
            ),
            "expected_priorities": expected_schedule,
            "predicted_priorities": predicted_schedule,
            "schedule_exact": expected_schedule == predicted_schedule,
            "schedule_intent_consistent": (
                scheduled_agents == predicted_intents
            ),
            "entity_correct": entity_correct,
            "entity_total": entity_total,
            "entity_checks": entity_checks,
            "latency_seconds": round(elapsed, 4),
            "attempt_count": attempt_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "error": error,
            "raw_prediction": prediction,
        })
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"intent={'PASS' if rows[-1]['intent_exact'] else 'FAIL'} "
            f"schedule={'PASS' if rows[-1]['schedule_exact'] else 'FAIL'} "
            f"{elapsed:.1f}s",
        )
        if args.delay:
            await asyncio.sleep(args.delay)

    expected_rows = [
        set(row["expected_intents"])
        for row in rows
    ]
    predicted_rows = [
        set(row["predicted_intents"])
        for row in rows
    ]
    intent_metrics = multilabel_classification(
        expected_rows,
        predicted_rows,
        INTENT_LABELS,
    )
    schedule_exact_count = sum(row["schedule_exact"] for row in rows)
    schedule_consistent_count = sum(
        row["schedule_intent_consistent"] for row in rows
    )
    entity_correct = sum(row["entity_correct"] for row in rows)
    entity_total = sum(row["entity_total"] for row in rows)
    error_count = sum(bool(row["error"]) for row in rows)
    rate_limit_retry_count = sum(
        row["rate_limit_retry_count"]
        for row in rows
    )

    report = {
        "evaluation_type": "intent",
        "generated_at": utc_now_iso(),
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(
            (PROJECT_ROOT / args.dataset).read_bytes(),
        ).hexdigest(),
        "model": LLM_CONFIG["model_name"],
        "thinking": args.thinking,
        "sample_count": len(rows),
        "metrics": {
            **intent_metrics,
            "schedule_exact_match": round(
                schedule_exact_count / len(rows) if rows else 0.0,
                4,
            ),
            "schedule_intent_consistency": round(
                schedule_consistent_count / len(rows) if rows else 0.0,
                4,
            ),
            "entity_field_accuracy": round(
                entity_correct / entity_total if entity_total else 0.0,
                4,
            ),
            "error_rate": round(
                error_count / len(rows) if rows else 0.0,
                4,
            ),
            "rate_limit_retry_count": rate_limit_retry_count,
        },
        "rows": rows,
        "caveats": [
            "Sample datasets are for pipeline validation, not resume claims.",
            "Relative dates can change with evaluation date.",
            "Run the final curated dataset at least three times to assess variance.",
        ],
    }
    destination = write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
