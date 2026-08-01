"""Run a paired, order-balanced sequential-vs-parallel latency benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agents.intention_agent import IntentionAgent
from agents.lazy_agent_registry import LazyAgentRegistry
from agents.orchestration_agent import OrchestrationAgent
from config import LLM_CONFIG
from config_agentscope import init_agentscope
from context.memory_manager import MemoryManager
from evaluation.common import (
    build_chat_model,
    load_jsonl,
    percentile,
    utc_now_iso,
    write_json,
)
from evaluation.metrics import latency_summary
from evaluation.run_latency_benchmark import (
    SequentialOrchestrationAgent,
    execute_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "evaluation/datasets/"
            "latency_orchestration.formal.v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "evaluation/reports/"
            "latency_orchestration_paired_v1.json"
        ),
    )
    parser.add_argument(
        "--progress",
        default=(
            "evaluation/reports/"
            "latency_orchestration_paired_v1.progress.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Cooldown between top-level requests; avoids turning latency "
        "measurement into a provider load test.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_map(schedule: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item.get("agent_name")): int(item.get("priority", 0))
        for item in schedule
    }


def summarize_values(values: list[float]) -> dict[str, float | None]:
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


def improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return round((baseline - candidate) / baseline, 4)


def build_runtime(
    *,
    mode: str,
    model: Any,
    storage_path: Path,
) -> tuple[IntentionAgent, OrchestrationAgent, LazyAgentRegistry]:
    memory = MemoryManager(
        user_id=f"latency_{mode}_user",
        session_id=f"latency_{mode}_session",
        storage_path=str(storage_path),
        llm_model=None,
    )
    memory.long_term.save_preference("hotel_brands", "如家")
    memory.long_term.save_preference("seat_preference", "靠过道")
    registry = LazyAgentRegistry(
        model=model,
        cache={},
        memory_manager=memory,
    )
    orchestrator_class = (
        OrchestrationAgent
        if mode == "parallel"
        else SequentialOrchestrationAgent
    )
    orchestrator = orchestrator_class(
        agent_registry=registry,
        memory_manager=memory,
    )
    intention_agent = IntentionAgent(
        name=f"LatencyIntentAgent_{mode}",
        model=model,
    )
    return intention_agent, orchestrator, registry


async def measured_query(
    *,
    case: dict[str, Any],
    repeat: int,
    pair_order: int,
    mode: str,
    intention_agent: IntentionAgent,
    orchestrator: OrchestrationAgent,
    registry: LazyAgentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    result: dict[str, Any] = {
        "status": "failure",
        "intent_seconds": 0.0,
        "intent_retry_count": 0,
        "intent_attempt_count": 0,
        "execution_seconds": 0.0,
        "total_seconds": 0.0,
        "agent_schedule": [],
        "agents_executed": 0,
        "orchestration_status": "not_started",
    }
    try:
        result = await execute_query(
            case["query"],
            intention_agent,
            orchestrator,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["total_seconds"] = round(time.perf_counter() - started, 4)

    expected_schedule = {
        str(name): int(priority)
        for name, priority in case["expected_schedule"].items()
    }
    actual_schedule = schedule_map(result.get("agent_schedule", []))
    schedule_matches = actual_schedule == expected_schedule
    return {
        "id": case["id"],
        "category": case["category"],
        "repeat": repeat,
        "pair_order": pair_order,
        "mode": mode,
        "query": case["query"],
        "expected_schedule": expected_schedule,
        "actual_schedule": actual_schedule,
        "schedule_matches_expected": schedule_matches,
        "loaded_agents_after_request": sorted(registry.get_loaded_agents()),
        **result,
        "error": error,
    }


async def warmup(
    *,
    cases: list[dict[str, Any]],
    rounds: int,
    mode: str,
    intention_agent: IntentionAgent,
    orchestrator: OrchestrationAgent,
) -> list[dict[str, Any]]:
    rows = []
    for warmup_round in range(1, rounds + 1):
        for case in cases:
            started = time.perf_counter()
            error = None
            status = "success"
            try:
                result = await execute_query(
                    case["query"],
                    intention_agent,
                    orchestrator,
                )
                status = result["status"]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = "failure"
            row = {
                "mode": mode,
                "warmup_round": warmup_round,
                "id": case["id"],
                "status": status,
                "seconds": round(time.perf_counter() - started, 4),
                "error": error,
            }
            rows.append(row)
            print(
                f"[warmup {mode} {warmup_round}] {case['id']} "
                f"{status} {row['seconds']:.1f}s",
                flush=True,
            )
    return rows


def mode_metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    metrics = latency_summary(selected)
    metrics["schedule_match_rate"] = round(
        sum(row["schedule_matches_expected"] for row in selected)
        / len(selected),
        4,
    )
    metrics["partial_failure_count"] = sum(
        row.get("orchestration_status") == "partial_failure"
        for row in selected
    )
    metrics["agent_retry_count"] = sum(
        int(row.get("agent_retry_count", 0))
        for row in selected
    )
    metrics["agent_error_count"] = sum(
        int(row.get("agent_error_count", 0))
        for row in selected
    )
    metrics["intent_retry_count"] = sum(
        int(row.get("intent_retry_count", 0))
        for row in selected
    )
    return metrics


def comparison_metrics(
    rows: list[dict[str, Any]],
    mode_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["id"], row["repeat"])][row["mode"]] = row

    complete_pairs = [
        pair for pair in paired.values()
        if set(pair) == {"sequential", "parallel"}
    ]
    total_pair_improvements = []
    execution_pair_improvements = []
    parallel_total_wins = 0
    parallel_execution_wins = 0
    for pair in complete_pairs:
        sequential = pair["sequential"]
        parallel = pair["parallel"]
        total_delta = improvement(
            sequential["total_seconds"],
            parallel["total_seconds"],
        )
        execution_delta = improvement(
            sequential["execution_seconds"],
            parallel["execution_seconds"],
        )
        if total_delta is not None:
            total_pair_improvements.append(total_delta)
        if execution_delta is not None:
            execution_pair_improvements.append(execution_delta)
        parallel_total_wins += (
            parallel["total_seconds"] < sequential["total_seconds"]
        )
        parallel_execution_wins += (
            parallel["execution_seconds"] < sequential["execution_seconds"]
        )

    baseline = mode_summaries["sequential"]
    candidate = mode_summaries["parallel"]
    aggregate = {}
    for phase in ("total_seconds", "execution_seconds", "intent_seconds"):
        aggregate[phase] = {
            metric: improvement(
                baseline[phase][metric],
                candidate[phase][metric],
            )
            for metric in ("mean", "p50", "p95")
        }

    return {
        "complete_pair_count": len(complete_pairs),
        "aggregate_improvement": aggregate,
        "paired_total_improvement": summarize_values(
            total_pair_improvements,
        ),
        "paired_execution_improvement": summarize_values(
            execution_pair_improvements,
        ),
        "parallel_total_win_rate": round(
            parallel_total_wins / len(complete_pairs),
            4,
        ),
        "parallel_execution_win_rate": round(
            parallel_execution_wins / len(complete_pairs),
            4,
        ),
        "success_rate_delta": round(
            candidate["success_rate"] - baseline["success_rate"],
            4,
        ),
    }


async def run() -> int:
    args = parse_args()
    if args.repeats < 1 or args.warmup_rounds < 1:
        raise ValueError("repeats and warmup-rounds must both be >= 1")
    dataset_path = PROJECT_ROOT / args.dataset
    cases = load_jsonl(dataset_path)
    for case in cases:
        if not case.get("expected_schedule"):
            raise ValueError(f"{case['id']}: expected_schedule is required")
        if len(case["expected_schedule"]) < 2:
            raise ValueError(f"{case['id']}: benchmark needs multiple agents")

    init_agentscope()
    model = build_chat_model(
        thinking_type=args.thinking,
        max_tokens=5000,
        temperature=0.1,
    )
    generated_at = utc_now_iso()
    rows = []
    warmup_rows = []
    progress_path = PROJECT_ROOT / args.progress

    with tempfile.TemporaryDirectory() as temp_root:
        temp_path = Path(temp_root)
        runtimes = {
            mode: build_runtime(
                mode=mode,
                model=model,
                storage_path=temp_path / mode,
            )
            for mode in ("sequential", "parallel")
        }

        for mode in ("sequential", "parallel"):
            intention_agent, orchestrator, _ = runtimes[mode]
            warmup_rows.extend(await warmup(
                cases=cases,
                rounds=args.warmup_rounds,
                mode=mode,
                intention_agent=intention_agent,
                orchestrator=orchestrator,
            ))
            write_json(progress_path, {
                "status": "warming_up",
                "updated_at": utc_now_iso(),
                "completed_warmup_requests": len(warmup_rows),
                "completed_measured_requests": 0,
                "expected_measured_requests": len(cases) * args.repeats * 2,
            })

        for repeat in range(1, args.repeats + 1):
            for case_index, case in enumerate(cases, 1):
                modes = (
                    ("sequential", "parallel")
                    if (repeat + case_index) % 2 == 0
                    else ("parallel", "sequential")
                )
                for pair_order, mode in enumerate(modes, 1):
                    intention_agent, orchestrator, registry = runtimes[mode]
                    row = await measured_query(
                        case=case,
                        repeat=repeat,
                        pair_order=pair_order,
                        mode=mode,
                        intention_agent=intention_agent,
                        orchestrator=orchestrator,
                        registry=registry,
                    )
                    rows.append(row)
                    write_json(progress_path, {
                        "status": "running",
                        "updated_at": utc_now_iso(),
                        "completed_warmup_requests": len(warmup_rows),
                        "completed_measured_requests": len(rows),
                        "expected_measured_requests": (
                            len(cases) * args.repeats * 2
                        ),
                        "last_row": {
                            "id": row["id"],
                            "repeat": row["repeat"],
                            "mode": row["mode"],
                            "status": row["status"],
                            "schedule_matches_expected": row[
                                "schedule_matches_expected"
                            ],
                            "total_seconds": row["total_seconds"],
                        },
                    })
                    print(
                        f"[{repeat}:{case_index} {mode}] {case['id']} "
                        f"{row['status']} schedule="
                        f"{'PASS' if row['schedule_matches_expected'] else 'FAIL'} "
                        f"total={row['total_seconds']:.1f}s "
                        f"exec={row['execution_seconds']:.1f}s",
                        flush=True,
                    )
                    if args.delay:
                        await asyncio.sleep(args.delay)

    mode_summaries = {
        mode: mode_metrics(rows, mode)
        for mode in ("sequential", "parallel")
    }
    comparison = comparison_metrics(rows, mode_summaries)
    expected_per_mode = len(cases) * args.repeats
    valid = (
        len([row for row in rows if row["mode"] == "sequential"])
        == expected_per_mode
        and len([row for row in rows if row["mode"] == "parallel"])
        == expected_per_mode
        and all(row["schedule_matches_expected"] for row in rows)
        and comparison["complete_pair_count"] == expected_per_mode
    )
    report = {
        "evaluation_type": "paired_orchestration_latency",
        "generated_at": generated_at,
        "completed_at": utc_now_iso(),
        "dataset": args.dataset,
        "dataset_sha256": sha256(dataset_path),
        "sample_count": len(cases),
        "repeats_per_mode": args.repeats,
        "measured_requests_per_mode": expected_per_mode,
        "warmup_rounds_per_mode": args.warmup_rounds,
        "model": LLM_CONFIG["model_name"],
        "thinking": args.thinking,
        "temperature": 0.1,
        "max_tokens": 5000,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "order_control": (
            "Sequential-first and parallel-first order alternates by "
            "repeat and case."
        ),
        "valid_benchmark": valid,
        "mode_metrics": mode_summaries,
        "comparison": comparison,
        "warmup_rows": warmup_rows,
        "rows": rows,
        "caveats": [
            "Provider and network variance still affect absolute latency.",
            "The paired order-balanced design reduces but cannot eliminate "
            "time-varying external load.",
            "End-to-end latency includes intent recognition, which is not "
            "parallelized; execution_seconds isolates orchestration.",
            "A latency reduction claim is allowed only when schedule matching "
            "is 100% and success rate does not materially regress.",
        ],
    }
    destination = write_json(PROJECT_ROOT / args.output, report)
    write_json(progress_path, {
        "status": "completed" if valid else "invalid",
        "updated_at": utc_now_iso(),
        "completed_warmup_requests": len(warmup_rows),
        "completed_measured_requests": len(rows),
        "expected_measured_requests": len(cases) * args.repeats * 2,
        "valid_benchmark": valid,
        "output": args.output,
    })
    print(json.dumps({
        "valid_benchmark": valid,
        "mode_metrics": mode_summaries,
        "comparison": comparison,
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
