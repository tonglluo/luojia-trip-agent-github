"""Run a frozen intent evaluation repeatedly and aggregate valid runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRIC_NAMES = (
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
        "--dataset",
        default="evaluation/datasets/intent_eval.holdout.v2.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/intent_holdout_v2_repeated",
    )
    parser.add_argument("--valid-runs", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(
        -(-len(ordered) * fraction // 1)
    ) - 1))
    return ordered[index]


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main() -> int:
    args = parse_args()
    if args.valid_runs < 1:
        raise SystemExit("--valid-runs must be positive")
    if args.max_attempts < args.valid_runs:
        raise SystemExit("--max-attempts must be >= --valid-runs")

    dataset_path = PROJECT_ROOT / args.dataset
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    sample_count = sum(
        bool(line.strip())
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    )
    progress_path = output_dir / "progress.json"

    valid_reports: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, args.max_attempts + 1):
        if len(valid_reports) >= args.valid_runs:
            break
        result_path = output_dir / f"attempt_{attempt}.json"
        log_path = output_dir / f"attempt_{attempt}.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "evaluation/run_intent_eval.py"),
            "--dataset",
            args.dataset,
            "--output",
            str(result_path.relative_to(PROJECT_ROOT)),
            "--thinking",
            args.thinking,
            "--delay",
            str(args.delay),
            "--max-retries",
            str(args.max_retries),
            "--retry-base-delay",
            str(args.retry_base_delay),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        report = load_json(result_path) if result_path.exists() else None
        valid = bool(
            report
            and report.get("dataset_sha256") == dataset_sha256
            and report.get("sample_count") == sample_count
            and report.get("metrics", {}).get("error_rate") == 0
        )
        attempt_record = {
            "attempt": attempt,
            "result": str(result_path.relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
            "process_exit_code": completed.returncode,
            "valid": valid,
            "error_rate": (
                report.get("metrics", {}).get("error_rate")
                if report
                else None
            ),
        }
        attempts.append(attempt_record)
        if valid and report:
            valid_reports.append(report)
        write_json(progress_path, {
            "status": (
                "complete"
                if len(valid_reports) >= args.valid_runs
                else "running"
            ),
            "dataset": args.dataset,
            "dataset_sha256": dataset_sha256,
            "required_valid_runs": args.valid_runs,
            "valid_run_count": len(valid_reports),
            "attempts": attempts,
        })

    if len(valid_reports) < args.valid_runs:
        write_json(progress_path, {
            "status": "insufficient_valid_runs",
            "dataset": args.dataset,
            "dataset_sha256": dataset_sha256,
            "required_valid_runs": args.valid_runs,
            "valid_run_count": len(valid_reports),
            "attempts": attempts,
        })
        raise SystemExit(
            f"Only {len(valid_reports)} valid runs after "
            f"{len(attempts)} attempts",
        )

    metric_values = {
        name: [
            float(report["metrics"][name])
            for report in valid_reports
        ]
        for name in METRIC_NAMES
    }
    all_latencies = [
        float(row["latency_seconds"])
        for report in valid_reports
        for row in report["rows"]
        if not row.get("error")
    ]
    per_run = []
    for index, report in enumerate(valid_reports, 1):
        latencies = [
            float(row["latency_seconds"])
            for row in report["rows"]
            if not row.get("error")
        ]
        per_run.append({
            "valid_run": index,
            "generated_at": report["generated_at"],
            "metrics": {
                name: report["metrics"][name]
                for name in METRIC_NAMES
            },
            "rate_limit_retry_count": report["metrics"].get(
                "rate_limit_retry_count",
                0,
            ),
            "latency": {
                "mean_seconds": round(statistics.fmean(latencies), 4),
                "p50_seconds": round(nearest_rank(latencies, 0.50), 4),
                "p95_seconds": round(nearest_rank(latencies, 0.95), 4),
                "max_seconds": round(max(latencies), 4),
            },
        })

    summary = {
        "evaluation_type": "intent_repeated",
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256,
        "model": valid_reports[0]["model"],
        "thinking": args.thinking,
        "sample_count_per_run": sample_count,
        "valid_run_count": len(valid_reports),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "metrics": {
            name: metric_summary(values)
            for name, values in metric_values.items()
        },
        "latency_all_valid_requests": {
            "request_count": len(all_latencies),
            "mean_seconds": round(
                statistics.fmean(all_latencies),
                4,
            ),
            "p50_seconds": round(
                nearest_rank(all_latencies, 0.50),
                4,
            ),
            "p95_seconds": round(
                nearest_rank(all_latencies, 0.95),
                4,
            ),
            "max_seconds": round(max(all_latencies), 4),
        },
        "per_run": per_run,
        "caveat": (
            "All attempts are retained. Only runs with matching dataset "
            "hash, complete sample count and zero API/parse errors are "
            "included in aggregate metrics."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
