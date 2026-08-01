"""Generate a claim-safe Markdown evaluation report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent",
        default="evaluation/reports/intent_results.json",
    )
    parser.add_argument(
        "--rag",
        default="evaluation/reports/rag_results.json",
    )
    parser.add_argument(
        "--latency",
        default="evaluation/reports/latency_results.json",
    )
    parser.add_argument(
        "--comparison",
        default="evaluation/reports/latency_comparison.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/evaluation_report.md",
    )
    return parser.parse_args()


def load_optional(path: str) -> dict | None:
    source = PROJECT_ROOT / path
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def manual_rag_mean(report: dict | None) -> float | None:
    if not report:
        return None
    scores = [
        row["manual_answer_score_0_to_2"]
        for row in report.get("rows", [])
        if row.get("manual_answer_score_0_to_2") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def main() -> int:
    args = parse_args()
    intent = load_optional(args.intent)
    rag = load_optional(args.rag)
    latency = load_optional(args.latency)
    comparison = load_optional(args.comparison)

    intent_count = intent.get("sample_count", 0) if intent else 0
    rag_count = rag.get("sample_count", 0) if rag else 0
    latency_count = (
        latency.get("metrics", {}).get("request_count", 0)
        if latency
        else 0
    )
    manual_mean = manual_rag_mean(rag)

    readiness_checks = {
        "Intent dataset has at least 90 reviewed cases": intent_count >= 90,
        "RAG dataset has at least 50 reviewed cases": rag_count >= 50,
        "All RAG answers have manual 0-2 scores": (
            bool(rag)
            and len([
                row
                for row in rag.get("rows", [])
                if row.get("manual_answer_score_0_to_2") is not None
            ])
            == rag_count
        ),
        "Latency has at least 30 measured requests": latency_count >= 30,
        "Latency comparison uses comparable settings": (
            bool(comparison) and comparison.get("comparable") is True
        ),
    }
    ready = all(readiness_checks.values())
    assessment = "Ready to share" if ready else "Needs revision"

    lines = [
        "# Aligo Evaluation Report",
        "",
        f"- Generated at: {utc_now_iso()}",
        f"- Overall assessment: **{assessment}**",
        "",
        "## Intent Recognition",
        "",
    ]
    if intent:
        metrics = intent["metrics"]
        lines.extend([
            f"- Samples: {intent_count}",
            f"- Exact Match: {percent(metrics.get('exact_match'))}",
            f"- Macro-F1: {percent(metrics.get('macro_f1'))}",
            (
                "- Schedule Exact Match: "
                f"{percent(metrics.get('schedule_exact_match'))}"
            ),
            (
                "- Schedule-Intent Consistency: "
                f"{percent(metrics.get('schedule_intent_consistency'))}"
            ),
            (
                "- Entity Field Accuracy: "
                f"{percent(metrics.get('entity_field_accuracy'))}"
            ),
            f"- Error Rate: {percent(metrics.get('error_rate'))}",
        ])
    else:
        lines.append("- Not evaluated.")

    lines.extend(["", "## RAG", ""])
    if rag:
        metrics = rag["metrics"]
        recall_key = next(
            (
                key
                for key in metrics
                if key.startswith("recall_at_")
            ),
            "recall_at_3",
        )
        lines.extend([
            f"- Samples: {rag_count}",
            f"- Backend: {rag.get('backend')}",
            f"- {recall_key}: {percent(metrics.get(recall_key))}",
            f"- MRR: {metrics.get('mrr', 'N/A')}",
            (
                "- Automated Key-point Coverage: "
                f"{percent(metrics.get('key_point_coverage'))}"
            ),
            (
                "- Unanswerable Hallucination Rate: "
                f"{percent(metrics.get('unanswerable_hallucination_rate'))}"
            ),
            (
                "- Manual Answer Score Mean (0-2): "
                f"{manual_mean:.2f}" if manual_mean is not None
                else "- Manual Answer Score Mean (0-2): N/A"
            ),
        ])
    else:
        lines.append("- Not evaluated.")

    lines.extend(["", "## Latency", ""])
    if latency:
        metrics = latency["metrics"]
        totals = metrics["total_seconds"]
        lines.extend([
            f"- Measured requests: {latency_count}",
            f"- Success Rate: {percent(metrics.get('success_rate'))}",
            f"- P50: {totals.get('p50')} seconds",
            f"- P95: {totals.get('p95')} seconds",
            f"- Mean: {totals.get('mean')} seconds",
            f"- Mode: {latency.get('execution_mode')}",
            f"- Thinking: {latency.get('thinking')}",
        ])
    else:
        lines.append("- Not evaluated.")

    if comparison:
        metrics = comparison["metrics"]
        lines.extend([
            "",
            "## Latency Comparison",
            "",
            f"- Comparable: {comparison.get('comparable')}",
            f"- P50 Improvement: {percent(metrics.get('p50_improvement'))}",
            f"- P95 Improvement: {percent(metrics.get('p95_improvement'))}",
            f"- Mean Improvement: {percent(metrics.get('mean_improvement'))}",
        ])

    lines.extend([
        "",
        "## Resume-Claim Readiness",
        "",
    ])
    for check, passed in readiness_checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {check}")

    lines.extend([
        "",
        "## Required Caveats",
        "",
        "- Sample datasets validate the pipeline but do not support headline claims.",
        "- RAG retrieval, automated key-point coverage, and manual answer correctness are different metrics.",
        "- Latency comparisons require the same model, dataset, network context, and measurement method.",
        "- Do not report README baseline numbers unless reproduced by these artifacts.",
        "",
    ])

    destination = PROJECT_ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {destination}")
    print(f"Assessment: {assessment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
