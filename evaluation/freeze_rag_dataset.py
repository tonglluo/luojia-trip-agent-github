"""Freeze a human-reviewed RAG dataset and write an immutable manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_rag_dataset import audit
from evaluation.build_rag_source_evidence import build as build_source_evidence
from evaluation.common import load_jsonl, utc_now_iso, write_json


APPROVAL_PHRASE = "I_REVIEWED_ALL_60_RAG"
DEFAULT_COMPARISON = "evaluation/datasets/rag_eval.sample.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="evaluation/datasets/rag_eval.formal.candidate.jsonl",
    )
    parser.add_argument(
        "--destination",
        default="evaluation/datasets/rag_eval.formal.v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/datasets/rag_eval.formal.v1.manifest.json",
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

    source = PROJECT_ROOT / args.source
    destination = PROJECT_ROOT / args.destination
    manifest_path = PROJECT_ROOT / args.manifest
    rows = load_jsonl(source)
    comparison_rows = load_jsonl(PROJECT_ROOT / DEFAULT_COMPARISON)
    audit_result = audit(
        rows,
        comparison_rows,
        dataset_path=source,
        minimum_answerable=50,
        minimum_unanswerable=10,
    )
    if audit_result["issues"] or audit_result["warnings"]:
        raise SystemExit(
            "Candidate audit has issues or warnings; refusing to freeze.",
        )
    evidence_result = build_source_evidence(
        rows,
        dataset_path=source,
    )
    if (
        evidence_result["missing_evidence_count"]
        or evidence_result["weak_evidence_count"]
    ):
        raise SystemExit(
            "Candidate has missing or weak line-level source evidence; "
            "refusing to freeze.",
        )
    if len(rows) != 60:
        raise SystemExit(f"Expected 60 reviewed rows, found {len(rows)}")
    if destination.exists() or manifest_path.exists():
        raise SystemExit(
            "Frozen dataset or manifest already exists; refusing to overwrite it.",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    manifest = {
        "dataset": args.destination,
        "sha256": sha256(destination),
        "sample_count": len(rows),
        "answerable_count": audit_result["answerable_count"],
        "unanswerable_count": audit_result["unanswerable_count"],
        "multi_source_count": audit_result["multi_source_count"],
        "answer_key_point_count": evidence_result[
            "answer_key_point_count"
        ],
        "line_level_evidence_coverage": evidence_result[
            "line_level_evidence_coverage"
        ],
        "source_counts": audit_result["source_counts"],
        "frozen_at": utc_now_iso(),
        "reviewer": args.reviewer,
        "approval_phrase": APPROVAL_PHRASE,
        "status": "frozen",
        "rules": [
            "Do not edit the frozen JSONL file.",
            "Do not change questions, expected sources or key points after evaluation.",
            "Create a new independently reviewed version for annotation changes.",
            "Keep retrieval and answer-generation metrics separate.",
        ],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
