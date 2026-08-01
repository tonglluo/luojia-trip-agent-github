"""Metric definitions used by the evaluation scripts."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Iterable

from evaluation.common import percentile, round_or_none


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(normalize_text(item) for item in value))
    text = str(value).strip().lower()
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", text)
    return "".join(text.split())


def searchable_text(value: Any) -> str:
    """Flatten nested model output for contains-based entity checks."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return "|".join(
            searchable_text(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple, set)):
        return "|".join(searchable_text(item) for item in value)
    return normalize_text(value)


def set_exact_match(expected: Iterable[str], predicted: Iterable[str]) -> bool:
    return set(expected) == set(predicted)


def multilabel_classification(
    expected_rows: list[set[str]],
    predicted_rows: list[set[str]],
    labels: Iterable[str],
) -> dict[str, Any]:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError("Expected and predicted row counts differ")

    per_label: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(labels)):
        tp = sum(
            label in expected and label in predicted
            for expected, predicted in zip(expected_rows, predicted_rows)
        )
        fp = sum(
            label not in expected and label in predicted
            for expected, predicted in zip(expected_rows, predicted_rows)
        )
        fn = sum(
            label in expected and label not in predicted
            for expected, predicted in zip(expected_rows, predicted_rows)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    exact_count = sum(
        expected == predicted
        for expected, predicted in zip(expected_rows, predicted_rows)
    )
    macro_f1 = (
        sum(item["f1"] for item in per_label.values()) / len(per_label)
        if per_label
        else 0.0
    )
    return {
        "sample_count": len(expected_rows),
        "exact_match": round(
            exact_count / len(expected_rows) if expected_rows else 0.0,
            4,
        ),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
    }


def schedule_to_priority_map(schedule: Iterable[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item.get("agent_name")): int(item.get("priority", 0))
        for item in schedule
        if item.get("agent_name")
    }


def entity_field_score(
    expected: dict[str, Any],
    predicted: dict[str, Any],
) -> tuple[int, int, dict[str, bool]]:
    checks: dict[str, bool] = {}
    for field, expected_value in expected.items():
        predicted_value = predicted.get(field)
        if isinstance(expected_value, dict) and "any_of" in expected_value:
            checks[field] = any(
                normalize_text(candidate) == normalize_text(predicted_value)
                for candidate in expected_value["any_of"]
            )
        elif (
            isinstance(expected_value, dict)
            and "contains_all" in expected_value
        ):
            predicted_text = searchable_text(predicted_value)
            checks[field] = all(
                normalize_text(candidate) in predicted_text
                for candidate in expected_value["contains_all"]
            )
        elif (
            isinstance(expected_value, dict)
            and "contains_any" in expected_value
        ):
            predicted_text = searchable_text(predicted_value)
            checks[field] = any(
                normalize_text(candidate) in predicted_text
                for candidate in expected_value["contains_any"]
            )
        elif (
            isinstance(expected_value, dict)
            and "contains_all_groups" in expected_value
        ):
            predicted_text = searchable_text(predicted_value)
            checks[field] = all(
                any(
                    normalize_text(candidate) in predicted_text
                    for candidate in alternatives
                )
                for alternatives in expected_value[
                    "contains_all_groups"
                ]
            )
        else:
            checks[field] = (
                normalize_text(expected_value)
                == normalize_text(predicted_value)
            )
    return sum(checks.values()), len(checks), checks


def retrieval_metrics(
    expected_sources_rows: list[list[str]],
    retrieved_sources_rows: list[list[str]],
    *,
    k: int,
) -> dict[str, Any]:
    any_hits = 0
    all_hits = 0
    source_recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    evaluated = 0

    for expected, retrieved in zip(
        expected_sources_rows,
        retrieved_sources_rows,
    ):
        expected_set = set(expected)
        if not expected_set:
            continue
        evaluated += 1
        top_sources = retrieved[:k]
        retrieved_set = set(top_sources)
        relevant_retrieved = expected_set & retrieved_set
        source_recalls.append(len(relevant_retrieved) / len(expected_set))
        if relevant_retrieved:
            any_hits += 1
        if expected_set <= retrieved_set:
            all_hits += 1
        rank = next(
            (
                index
                for index, source in enumerate(top_sources, 1)
                if source in expected_set
            ),
            None,
        )
        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "answerable_count": evaluated,
        # Kept for compatibility: historically recall_at_k meant any-source hit.
        f"recall_at_{k}": round(
            any_hits / evaluated if evaluated else 0.0,
            4,
        ),
        f"any_source_hit_at_{k}": round(
            any_hits / evaluated if evaluated else 0.0,
            4,
        ),
        f"all_sources_recalled_at_{k}": round(
            all_hits / evaluated if evaluated else 0.0,
            4,
        ),
        f"mean_source_recall_at_{k}": round(
            sum(source_recalls) / evaluated if evaluated else 0.0,
            4,
        ),
        "mrr": round(
            sum(reciprocal_ranks) / evaluated if evaluated else 0.0,
            4,
        ),
    }


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    totals = [row["total_seconds"] for row in successful]
    intent = [row["intent_seconds"] for row in successful]
    execution = [row["execution_seconds"] for row in successful]

    def summarize(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": round_or_none(
                sum(values) / len(values) if values else None,
            ),
            "p50": round_or_none(percentile(values, 0.50)),
            "p95": round_or_none(percentile(values, 0.95)),
            "min": round_or_none(min(values) if values else None),
            "max": round_or_none(max(values) if values else None),
        }

    return {
        "request_count": len(rows),
        "success_count": len(successful),
        "success_rate": round(
            len(successful) / len(rows) if rows else 0.0,
            4,
        ),
        "total_seconds": summarize(totals),
        "intent_seconds": summarize(intent),
        "execution_seconds": summarize(execution),
    }
