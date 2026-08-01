"""Create a compact diagnostic summary from an intent evaluation report."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import percentile, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/reports/intent_results.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_diagnostics.json",
    )
    return parser.parse_args()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "p50": round(float(percentile(values, 0.50)), 4),
        "p95": round(float(percentile(values, 0.95)), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows", [])
    missing_intents: Counter[str] = Counter()
    extra_intents: Counter[str] = Counter()
    missing_agents: Counter[str] = Counter()
    extra_agents: Counter[str] = Counter()
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for row in rows:
        expected_intents = set(row.get("expected_intents", []))
        predicted_intents = set(row.get("predicted_intents", []))
        expected_schedule = set(row.get("expected_priorities", {}))
        predicted_schedule = set(row.get("predicted_priorities", {}))
        missing_intents.update(expected_intents - predicted_intents)
        extra_intents.update(predicted_intents - expected_intents)
        missing_agents.update(expected_schedule - predicted_schedule)
        extra_agents.update(predicted_schedule - expected_schedule)
        categories[str(row.get("category") or "uncategorized")].append(row)

        entity_failures = [
            field
            for field, passed in row.get("entity_checks", {}).items()
            if not passed
        ]
        if (
            not row.get("intent_exact")
            or not row.get("schedule_exact")
            or entity_failures
            or row.get("error")
        ):
            failures.append({
                "id": row.get("id"),
                "category": row.get("category"),
                "query": row.get("query"),
                "missing_intents": sorted(expected_intents - predicted_intents),
                "extra_intents": sorted(predicted_intents - expected_intents),
                "missing_scheduled_agents": sorted(
                    expected_schedule - predicted_schedule,
                ),
                "extra_scheduled_agents": sorted(
                    predicted_schedule - expected_schedule,
                ),
                "priority_mismatch": {
                    agent: {
                        "expected": row.get("expected_priorities", {}).get(agent),
                        "predicted": row.get("predicted_priorities", {}).get(agent),
                    }
                    for agent in expected_schedule & predicted_schedule
                    if row.get("expected_priorities", {}).get(agent)
                    != row.get("predicted_priorities", {}).get(agent)
                },
                "entity_failures": entity_failures,
                "predicted_entities": (
                    row.get("raw_prediction", {}).get("key_entities", {})
                ),
                "latency_seconds": row.get("latency_seconds"),
                "error": row.get("error"),
            })

    category_metrics = {}
    for category, category_rows in sorted(categories.items()):
        entity_total = sum(row.get("entity_total", 0) for row in category_rows)
        entity_correct = sum(
            row.get("entity_correct", 0) for row in category_rows
        )
        category_metrics[category] = {
            "count": len(category_rows),
            "intent_exact_match": _rate(
                sum(bool(row.get("intent_exact")) for row in category_rows),
                len(category_rows),
            ),
            "schedule_exact_match": _rate(
                sum(bool(row.get("schedule_exact")) for row in category_rows),
                len(category_rows),
            ),
            "entity_field_accuracy": _rate(entity_correct, entity_total),
        }

    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") is not None and not row.get("error")
    ]
    slowest = sorted(
        (
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "query": row.get("query"),
                "latency_seconds": row.get("latency_seconds"),
            }
            for row in rows
            if row.get("latency_seconds") is not None
        ),
        key=lambda item: float(item["latency_seconds"]),
        reverse=True,
    )[:5]

    return {
        "generated_at": utc_now_iso(),
        "source_report": str(report.get("dataset", "")),
        "sample_count": len(rows),
        "headline_metrics": report.get("metrics", {}),
        "intent_error_counts": {
            "missing": dict(missing_intents.most_common()),
            "extra": dict(extra_intents.most_common()),
        },
        "schedule_error_counts": {
            "missing": dict(missing_agents.most_common()),
            "extra": dict(extra_agents.most_common()),
        },
        "category_metrics": category_metrics,
        "latency_seconds": _latency_summary(latencies),
        "slowest_cases": slowest,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    source = PROJECT_ROOT / args.input
    report = json.loads(source.read_text(encoding="utf-8"))
    diagnostics = analyze(report)
    destination = write_json(PROJECT_ROOT / args.output, diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
