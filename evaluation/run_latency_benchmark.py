"""Benchmark end-to-end intent recognition and agent orchestration latency."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agentscope.message import Msg

from agents.intention_agent import IntentionAgent
from agents.lazy_agent_registry import LazyAgentRegistry
from agents.orchestration_agent import OrchestrationAgent
from config import LLM_CONFIG, RESILIENCE_CONFIG
from config_agentscope import init_agentscope
from context.memory_manager import MemoryManager
from evaluation.common import (
    build_chat_model,
    load_jsonl,
    utc_now_iso,
    write_json,
)
from evaluation.metrics import latency_summary
from utils.llm_resilience import retry_with_backoff


class SequentialOrchestrationAgent(OrchestrationAgent):
    """Benchmark-only variant that disables same-priority concurrency."""

    async def _execute_parallel_agents(
        self,
        tasks,
        context,
        previous_results,
    ):
        results = []
        for task in tasks:
            result = await self._execute_agent(
                agent_name=task.get("agent_name"),
                context=context,
                reason=task.get("reason", ""),
                expected_output=task.get("expected_output", ""),
                previous_results=previous_results,
            )
            results.append({
                "agent_name": task.get("agent_name"),
                "priority": task.get("priority", 0),
                "result": result,
            })
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/latency_queries.sample.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/latency_results.json",
    )
    parser.add_argument(
        "--label",
        default="benchmark",
        help="Experiment label, e.g. parallel_disabled_thinking.",
    )
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential"],
        default="parallel",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.5)
    return parser.parse_args()


async def execute_query(
    query: str,
    intention_agent: IntentionAgent,
    orchestrator: OrchestrationAgent,
) -> dict:
    total_started = time.perf_counter()
    intent_started = time.perf_counter()
    intent_attempt_count = 0

    async def call_intention():
        nonlocal intent_attempt_count
        intent_attempt_count += 1
        return await intention_agent.reply(
            Msg(name="user", role="user", content=query),
        )

    intention_response = await retry_with_backoff(
        call_intention,
        max_retries=int(RESILIENCE_CONFIG.get("max_retries", 3)),
        base_delay_sec=float(
            RESILIENCE_CONFIG.get("retry_base_delay_sec", 1.0),
        ),
        max_delay_sec=float(
            RESILIENCE_CONFIG.get("retry_max_delay_sec", 30.0),
        ),
    )
    intent_seconds = time.perf_counter() - intent_started
    intention_payload = json.loads(intention_response.content)

    execution_started = time.perf_counter()
    orchestration_response = await orchestrator.reply(intention_response)
    execution_seconds = time.perf_counter() - execution_started
    orchestration_payload = json.loads(orchestration_response.content)

    return {
        "status": (
            "success"
            if orchestration_payload.get("status") == "completed"
            else "failure"
        ),
        "intent_seconds": round(intent_seconds, 4),
        "intent_retry_count": intent_attempt_count - 1,
        "intent_attempt_count": intent_attempt_count,
        "execution_seconds": round(execution_seconds, 4),
        "total_seconds": round(
            time.perf_counter() - total_started,
            4,
        ),
        "predicted_intents": [
            item.get("type")
            for item in intention_payload.get("intents", [])
        ],
        "agent_schedule": intention_payload.get("agent_schedule", []),
        "agents_executed": orchestration_payload.get(
            "agents_executed",
            0,
        ),
        "agent_retry_count": orchestration_payload.get(
            "agent_retry_count",
            0,
        ),
        "agent_error_count": orchestration_payload.get("errors", 0),
        "orchestration_status": orchestration_payload.get("status"),
    }


async def run() -> int:
    args = parse_args()
    cases = load_jsonl(PROJECT_ROOT / args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0")

    init_agentscope()
    model = build_chat_model(
        thinking_type=args.thinking,
        max_tokens=5000,
        temperature=0.1,
    )
    rows = []

    with tempfile.TemporaryDirectory() as storage_path:
        memory = MemoryManager(
            user_id="latency_eval_user",
            session_id="latency_eval_session",
            storage_path=storage_path,
            llm_model=None,
        )
        registry = LazyAgentRegistry(
            model=model,
            cache={},
            memory_manager=memory,
        )
        orchestrator_class = (
            OrchestrationAgent
            if args.mode == "parallel"
            else SequentialOrchestrationAgent
        )
        orchestrator = orchestrator_class(
            agent_registry=registry,
            memory_manager=memory,
        )
        intention_agent = IntentionAgent(
            name="LatencyIntentAgent",
            model=model,
        )

        for warmup_index in range(args.warmup):
            case = cases[warmup_index % len(cases)]
            print(f"Warmup {warmup_index + 1}/{args.warmup}: {case['id']}")
            try:
                await execute_query(
                    case["query"],
                    intention_agent,
                    orchestrator,
                )
            except Exception as exc:
                print(f"Warmup failed: {exc}")

        for repeat in range(1, args.repeats + 1):
            for index, case in enumerate(cases, 1):
                error = None
                result = {
                    "status": "failure",
                    "intent_seconds": 0.0,
                    "intent_retry_count": 0,
                    "intent_attempt_count": 0,
                    "execution_seconds": 0.0,
                    "total_seconds": 0.0,
                }
                started_at = time.perf_counter()
                try:
                    result = await execute_query(
                        case["query"],
                        intention_agent,
                        orchestrator,
                    )
                except Exception as exc:
                    error = str(exc)
                    result["total_seconds"] = round(
                        time.perf_counter() - started_at,
                        4,
                    )

                row = {
                    "id": case["id"],
                    "category": case.get("category"),
                    "repeat": repeat,
                    "query": case["query"],
                    **result,
                    "error": error,
                }
                rows.append(row)
                print(
                    f"[{repeat}:{index}/{len(cases)}] {case['id']} "
                    f"{row['status']} total={row['total_seconds']:.1f}s",
                )
                if args.delay:
                    await asyncio.sleep(args.delay)

    report = {
        "evaluation_type": "latency",
        "generated_at": utc_now_iso(),
        "label": args.label,
        "dataset": args.dataset,
        "model": LLM_CONFIG["model_name"],
        "thinking": args.thinking,
        "execution_mode": args.mode,
        "warmup_count": args.warmup,
        "repeats": args.repeats,
        "metrics": latency_summary(rows),
        "rows": rows,
        "caveats": [
            "Network and provider load can affect latency.",
            "Compare runs only when model, dataset, and environment match.",
            "Use at least 30 measured requests before resume claims.",
        ],
    }
    destination = write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
