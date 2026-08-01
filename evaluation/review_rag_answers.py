"""Create or finalize a separate human-review file for RAG answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_jsonl, utc_now_iso, write_json


VALID_UNANSWERABLE_OUTCOMES = {
    "correct_refusal",
    "hallucination",
    "ambiguous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="evaluation/reports/rag_formal_v1_answers.json",
    )
    parser.add_argument(
        "--scores",
        default="evaluation/reports/rag_formal_v1_manual_review.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_v1_final.json",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a non-overwriting JSONL review template.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_template(report: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise SystemExit(f"Review file already exists; refusing to overwrite: {path}")
    rows = []
    for row in report["rows"]:
        answerable = bool(row["answerable"])
        rows.append({
            "id": row["id"],
            "question": row["question"],
            "answerable": answerable,
            "expected_sources": row["expected_sources"],
            "answer_key_points": row["answer_key_points"],
            "answer": row["answer"],
            "answer_score_0_to_2": None,
            "unanswerable_outcome": None,
            "review_note": "",
            "instructions": (
                "可回答题填写0/1/2；不可回答题保持分数为空，并填写"
                "correct_refusal、hallucination或ambiguous。"
            ),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize(
    report: dict[str, Any],
    review_rows: list[dict[str, Any]],
    *,
    report_path: Path,
    scores_path: Path,
) -> dict[str, Any]:
    report_rows = {row["id"]: row for row in report["rows"]}
    reviewed = {row["id"]: row for row in review_rows}
    if set(reviewed) != set(report_rows):
        missing = sorted(set(report_rows) - set(reviewed))
        extra = sorted(set(reviewed) - set(report_rows))
        raise SystemExit(
            f"Review IDs do not match report; missing={missing}, extra={extra}",
        )

    answer_scores: list[int] = []
    outcome_counts = {
        "correct_refusal": 0,
        "hallucination": 0,
        "ambiguous": 0,
    }
    merged_rows = []
    for row_id, report_row in report_rows.items():
        review = reviewed[row_id]
        if bool(review.get("answerable")) != bool(report_row["answerable"]):
            raise SystemExit(f"{row_id}: answerable flag differs from report")

        score = review.get("answer_score_0_to_2")
        outcome = review.get("unanswerable_outcome")
        if report_row["answerable"]:
            if type(score) is not int or score not in {0, 1, 2}:
                raise SystemExit(
                    f"{row_id}: answerable case needs integer score 0, 1 or 2",
                )
            if outcome is not None:
                raise SystemExit(
                    f"{row_id}: answerable case must not have refusal outcome",
                )
            answer_scores.append(score)
        else:
            if score is not None:
                raise SystemExit(
                    f"{row_id}: unanswerable case must not have answer score",
                )
            if outcome not in VALID_UNANSWERABLE_OUTCOMES:
                raise SystemExit(
                    f"{row_id}: invalid unanswerable outcome {outcome!r}",
                )
            outcome_counts[outcome] += 1

        merged_rows.append({
            **report_row,
            "manual_answer_score_0_to_2": score,
            "manual_unanswerable_outcome": outcome,
            "manual_review_note": review.get("review_note", ""),
        })

    answer_score_mean = round(sum(answer_scores) / len(answer_scores), 4)
    unanswerable_count = sum(outcome_counts.values())
    correct_refusal_rate = round(
        outcome_counts["correct_refusal"] / unanswerable_count,
        4,
    )
    return {
        "evaluation_type": "rag_final_human_reviewed",
        "generated_at": utc_now_iso(),
        "source_report": str(report_path.relative_to(PROJECT_ROOT)),
        "source_report_sha256": sha256(report_path),
        "dataset": report["dataset"],
        "dataset_sha256": report["dataset_sha256"],
        "model": report["model"],
        "thinking": report["thinking"],
        "sample_count": report["sample_count"],
        "metrics": {
            **report["metrics"],
            "manual_answer_score_mean": answer_score_mean,
            "manual_correct_refusal_rate": correct_refusal_rate,
            "manual_unanswerable_outcomes": outcome_counts,
        },
        "threshold_checks": {
            "recall_at_5": {
                "actual": report["metrics"].get("recall_at_5"),
                "threshold": 0.90,
                "passed": (
                    report["metrics"].get("recall_at_5") is not None
                    and report["metrics"]["recall_at_5"] >= 0.90
                ),
            },
            "manual_answer_score_mean": {
                "actual": answer_score_mean,
                "threshold": 1.60,
                "passed": answer_score_mean >= 1.60,
            },
            "manual_correct_refusal_rate": {
                "actual": correct_refusal_rate,
                "threshold": 0.90,
                "passed": correct_refusal_rate >= 0.90,
            },
        },
        "manual_review_file": str(scores_path.relative_to(PROJECT_ROOT)),
        "manual_review_sha256": sha256(scores_path),
        "rows": merged_rows,
        "caveats": [
            "Manual scores are stored separately from raw model output.",
            "Keyword coverage is diagnostic and not a semantic score.",
            "Ambiguous unanswerable outcomes count as not correctly refused.",
        ],
    }


def main() -> int:
    args = parse_args()
    report_path = PROJECT_ROOT / args.report
    scores_path = PROJECT_ROOT / args.scores
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if args.create_template:
        create_template(report, scores_path)
        print(f"Created review template: {scores_path}")
        return 0

    review_rows = load_jsonl(scores_path)
    result = finalize(
        report,
        review_rows,
        report_path=report_path,
        scores_path=scores_path,
    )
    destination = write_json(PROJECT_ROOT / args.output, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0 if all(
        item["passed"] for item in result["threshold_checks"].values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
