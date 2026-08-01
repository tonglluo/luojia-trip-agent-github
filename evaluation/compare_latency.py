"""Compare two latency reports with strict comparability checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--output",
        default="evaluation/reports/latency_comparison.json",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow model/dataset/thinking mismatches, with caveats.",
    )
    return parser.parse_args()


def load_report(path: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / path).read_text(encoding="utf-8"),
    )


def improvement(baseline: float | None, candidate: float | None):
    if baseline in (None, 0) or candidate is None:
        return None
    return round((baseline - candidate) / baseline, 4)


def main() -> int:
    args = parse_args()
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)

    comparability_fields = ["dataset", "model", "thinking"]
    mismatches = {
        field: {
            "baseline": baseline.get(field),
            "candidate": candidate.get(field),
        }
        for field in comparability_fields
        if baseline.get(field) != candidate.get(field)
    }
    if mismatches and not args.allow_mismatch:
        raise ValueError(
            "Reports are not directly comparable: "
            + json.dumps(mismatches, ensure_ascii=False),
        )

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    comparison = {
        "evaluation_type": "latency_comparison",
        "generated_at": utc_now_iso(),
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "comparable": not mismatches,
        "mismatches": mismatches,
        "metrics": {
            "p50_improvement": improvement(
                baseline_metrics["total_seconds"]["p50"],
                candidate_metrics["total_seconds"]["p50"],
            ),
            "p95_improvement": improvement(
                baseline_metrics["total_seconds"]["p95"],
                candidate_metrics["total_seconds"]["p95"],
            ),
            "mean_improvement": improvement(
                baseline_metrics["total_seconds"]["mean"],
                candidate_metrics["total_seconds"]["mean"],
            ),
            "baseline_success_rate": baseline_metrics["success_rate"],
            "candidate_success_rate": candidate_metrics["success_rate"],
        },
        "required_caveat": (
            "Runs have mismatched settings and cannot support a causal claim."
            if mismatches
            else None
        ),
    }
    destination = write_json(PROJECT_ROOT / args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
