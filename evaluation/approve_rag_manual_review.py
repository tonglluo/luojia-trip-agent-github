"""Convert explicitly approved AI suggestions into human review evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso, write_json


APPROVAL_PHRASE = "I_APPROVE_RAG_V1_SCORES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="evaluation/reports/rag_formal_v1_answers_attempt3.json",
    )
    parser.add_argument(
        "--pre-review",
        default=(
            "evaluation/reports/"
            "rag_formal_v1_grounded_pre_review.json"
        ),
    )
    parser.add_argument(
        "--destination",
        default=(
            "evaluation/reports/"
            "rag_formal_v1_manual_review.approved.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "evaluation/reports/"
            "rag_formal_v1_manual_review.approved.manifest.json"
        ),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.approval != APPROVAL_PHRASE:
        raise SystemExit(
            f"Approval phrase must be exactly: {APPROVAL_PHRASE}",
        )
    report_path = PROJECT_ROOT / args.report
    pre_review_path = PROJECT_ROOT / args.pre_review
    destination = PROJECT_ROOT / args.destination
    manifest_path = PROJECT_ROOT / args.manifest
    if destination.exists() or manifest_path.exists():
        raise SystemExit(
            "Approved review or manifest already exists; refusing to overwrite.",
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    pre_review = json.loads(pre_review_path.read_text(encoding="utf-8"))
    if (
        pre_review["review_type"]
        != "evidence_grounded_ai_pre_review_not_human"
    ):
        raise SystemExit(
            "Only the evidence-grounded pre-review may be approved",
        )
    if report["metrics"]["error_rate"] != 0:
        raise SystemExit("Source answer report is not valid")
    if report["dataset_sha256"] != pre_review["dataset_sha256"]:
        raise SystemExit("Pre-review and answer report dataset hashes differ")
    report_rows = {row["id"]: row for row in report["rows"]}
    review_rows = {row["id"]: row for row in pre_review["rows"]}
    if set(report_rows) != set(review_rows) or len(report_rows) != 60:
        raise SystemExit("Pre-review IDs do not match the 60 answer rows")

    approved_at = utc_now_iso()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row_id, answer_row in report_rows.items():
            suggested = review_rows[row_id]
            answerable = bool(answer_row["answerable"])
            approved_row = {
                "id": row_id,
                "question": answer_row["question"],
                "answerable": answerable,
                "expected_sources": answer_row["expected_sources"],
                "answer_key_points": answer_row["answer_key_points"],
                "answer": answer_row["answer"],
                "answer_score_0_to_2": (
                    suggested["suggested_answer_score_0_to_2"]
                    if answerable
                    else None
                ),
                "unanswerable_outcome": (
                    None
                    if answerable
                    else suggested["suggested_unanswerable_outcome"]
                ),
                "review_note": suggested["review_rationale"],
                "reviewer": args.reviewer,
                "approved_at": approved_at,
            }
            file.write(json.dumps(approved_row, ensure_ascii=False) + "\n")

    manifest = {
        "review_file": args.destination,
        "review_sha256": sha256(destination),
        "source_answer_report": args.report,
        "source_answer_report_sha256": sha256(report_path),
        "pre_review": args.pre_review,
        "pre_review_sha256": sha256(pre_review_path),
        "dataset_sha256": report["dataset_sha256"],
        "sample_count": len(report_rows),
        "reviewer": args.reviewer,
        "approved_at": approved_at,
        "approval_phrase": APPROVAL_PHRASE,
        "status": "human_approved",
        "note": (
            "The reviewer explicitly approved the AI-assisted suggestions "
            "after reviewing the worksheet and may not be inferred from "
            "automatic metrics alone."
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
