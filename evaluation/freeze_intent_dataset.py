"""Freeze a human-reviewed intent dataset and write its immutable manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_jsonl, utc_now_iso, write_json


APPROVAL_PHRASE = "I_REVIEWED_ALL_60"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="evaluation/datasets/intent_eval.holdout.candidate.jsonl",
    )
    parser.add_argument(
        "--destination",
        default="evaluation/datasets/intent_eval.holdout.v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/datasets/intent_eval.holdout.v1.manifest.json",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
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
        "sha256": _sha256(destination),
        "sample_count": len(rows),
        "frozen_at": utc_now_iso(),
        "reviewer": args.reviewer,
        "approval_phrase": APPROVAL_PHRASE,
        "status": "frozen",
        "rules": [
            "Do not edit the frozen JSONL file.",
            "Create a new version when annotations must change.",
            "Run evaluation before inspecting or changing model rules.",
        ],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
