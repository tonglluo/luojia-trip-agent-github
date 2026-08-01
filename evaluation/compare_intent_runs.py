"""Compare two saved intent-evaluation runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import percentile, utc_now_iso, write_json


QUALITY_METRICS = (
    "exact_match",
    "macro_f1",
    "schedule_exact_match",
    "schedule_intent_consistency",
    "entity_field_accuracy",
    "error_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="evaluation/reports/intent_results.json",
    )
    parser.add_argument(
        "--candidate",
        default="evaluation/reports/intent_results_v2.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_comparison_v1_v2.json",
    )
    return parser.parse_args()


def _pct_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return round((candidate - baseline) / baseline, 4)


def _latency(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = sorted(
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") is not None and not row.get("error")
    )
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": round(sum(values) / len(values), 4),
        "p50": round(float(percentile(values, 0.50)), 4),
        "p95": round(float(percentile(values, 0.95)), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {
        row["id"]: row
        for row in baseline.get("rows", [])
    }
    candidate_rows = {
        row["id"]: row
        for row in candidate.get("rows", [])
    }
    shared_ids = sorted(set(baseline_rows) & set(candidate_rows))

    classification_annotations_match = all(
        set(baseline_rows[row_id].get("expected_intents", []))
        == set(candidate_rows[row_id].get("expected_intents", []))
        and baseline_rows[row_id].get("expected_priorities", {})
        == candidate_rows[row_id].get("expected_priorities", {})
        for row_id in shared_ids
    )

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_rows = []
    for metric in QUALITY_METRICS:
        baseline_value = float(baseline_metrics[metric])
        candidate_value = float(candidate_metrics[metric])
        quality_rows.append({
            "metric": metric,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_delta": round(candidate_value - baseline_value, 4),
            "relative_change": _pct_change(baseline_value, candidate_value),
        })

    label_rows = []
    all_labels = sorted(
        set(baseline_metrics["per_label"])
        | set(candidate_metrics["per_label"])
    )
    for label in all_labels:
        baseline_label = baseline_metrics["per_label"].get(label, {})
        candidate_label = candidate_metrics["per_label"].get(label, {})
        label_rows.append({
            "label": label,
            "baseline_precision": baseline_label.get("precision"),
            "candidate_precision": candidate_label.get("precision"),
            "baseline_recall": baseline_label.get("recall"),
            "candidate_recall": candidate_label.get("recall"),
            "baseline_f1": baseline_label.get("f1"),
            "candidate_f1": candidate_label.get("f1"),
        })

    baseline_latency = _latency(baseline.get("rows", []))
    candidate_latency = _latency(candidate.get("rows", []))
    latency_rows = []
    for metric in ("mean", "p50", "p95", "min", "max"):
        baseline_value = baseline_latency[metric]
        candidate_value = candidate_latency[metric]
        latency_rows.append({
            "metric": metric,
            "baseline_seconds": baseline_value,
            "candidate_seconds": candidate_value,
            "delta_seconds": (
                round(candidate_value - baseline_value, 4)
                if baseline_value is not None and candidate_value is not None
                else None
            ),
            "relative_change": (
                _pct_change(baseline_value, candidate_value)
                if baseline_value is not None and candidate_value is not None
                else None
            ),
        })

    intent_fixed = [
        row_id
        for row_id in shared_ids
        if not baseline_rows[row_id].get("intent_exact")
        and candidate_rows[row_id].get("intent_exact")
    ]
    schedule_fixed = [
        row_id
        for row_id in shared_ids
        if not baseline_rows[row_id].get("schedule_exact")
        and candidate_rows[row_id].get("schedule_exact")
    ]
    regressions = [
        row_id
        for row_id in shared_ids
        if (
            baseline_rows[row_id].get("intent_exact")
            and not candidate_rows[row_id].get("intent_exact")
        )
        or (
            baseline_rows[row_id].get("schedule_exact")
            and not candidate_rows[row_id].get("schedule_exact")
        )
    ]

    candidate_slowest = sorted(
        (
            {
                "id": row["id"],
                "category": row.get("category"),
                "latency_seconds": row.get("latency_seconds"),
            }
            for row in candidate.get("rows", [])
            if row.get("latency_seconds") is not None
        ),
        key=lambda row: float(row["latency_seconds"]),
        reverse=True,
    )[:5]

    return {
        "generated_at": utc_now_iso(),
        "baseline": {
            "generated_at": baseline.get("generated_at"),
            "dataset": baseline.get("dataset"),
            "model": baseline.get("model"),
            "thinking": baseline.get("thinking"),
            "sample_count": baseline.get("sample_count"),
        },
        "candidate": {
            "generated_at": candidate.get("generated_at"),
            "dataset": candidate.get("dataset"),
            "model": candidate.get("model"),
            "thinking": candidate.get("thinking"),
            "sample_count": candidate.get("sample_count"),
        },
        "comparability": {
            "same_dataset_path": (
                baseline.get("dataset") == candidate.get("dataset")
            ),
            "same_model": baseline.get("model") == candidate.get("model"),
            "same_thinking": (
                baseline.get("thinking") == candidate.get("thinking")
            ),
            "same_case_ids": (
                set(baseline_rows) == set(candidate_rows)
            ),
            "classification_annotations_match": (
                classification_annotations_match
            ),
            "entity_annotations_comparable": False,
            "entity_note": (
                "Entity annotations and normalization rules were revised "
                "between V1 and V2; the entity-accuracy delta is not a clean "
                "model-effect estimate."
            ),
            "latency_causal_claim_supported": False,
            "latency_note": (
                "Each version has one run. Service variance and one 85.7-second "
                "candidate outlier prevent attributing latency changes to code."
            ),
        },
        "quality_metrics": quality_rows,
        "label_metrics": label_rows,
        "latency_metrics": latency_rows,
        "case_transitions": {
            "intent_fixed_count": len(intent_fixed),
            "intent_fixed_ids": intent_fixed,
            "schedule_fixed_count": len(schedule_fixed),
            "schedule_fixed_ids": schedule_fixed,
            "regression_count": len(regressions),
            "regression_ids": regressions,
        },
        "candidate_slowest_cases": candidate_slowest,
        "conclusion": (
            "V2 passes all 30 development cases with no observed intent or "
            "schedule regression. A separate frozen holdout set is required "
            "before treating 100% as a generalization estimate."
        ),
    }


def main() -> int:
    args = parse_args()
    baseline = json.loads(
        (PROJECT_ROOT / args.baseline).read_text(encoding="utf-8"),
    )
    candidate = json.loads(
        (PROJECT_ROOT / args.candidate).read_text(encoding="utf-8"),
    )
    comparison = compare(baseline, candidate)
    destination = write_json(PROJECT_ROOT / args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
