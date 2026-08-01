"""Evaluate RAG retrieval and optionally generated-answer quality."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.message import Msg

from agents.lazy_agent_registry import LazyAgentRegistry
from config import LLM_CONFIG
from config_agentscope import init_agentscope
from evaluation.common import (
    build_chat_model,
    load_jsonl,
    utc_now_iso,
    write_json,
)
from evaluation.metrics import normalize_text, retrieval_metrics


REFUSAL_PHRASES = [
    "没有找到相关信息",
    "知识库中没有",
    "无法根据知识库",
    "没有包含",
    "无法回答",
    "知识库未提供",
    "知识库中未提及",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/rag_eval.sample.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_results.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Call the LLM and score key-point coverage/refusal behavior.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
    )
    return parser.parse_args()


def retry_reason(value: object) -> str | None:
    message = str(value).casefold()
    if (
        "429" in message
        or "rate limit" in message
        or "速率限制" in message
        or "'code': '1302'" in message
        or '"code": "1302"' in message
    ):
        return "rate_limit"
    if (
        "timeout" in message
        or "timed out" in message
        or "request timed out" in message
    ):
        return "timeout"
    if any(
        marker in message
        for marker in ("500", "502", "503", "504")
    ):
        return "server_error"
    if any(
        marker in message
        for marker in (
            "connection error",
            "connection reset",
            "connection refused",
            "network error",
        )
    ):
        return "connection_error"
    return None


def key_point_coverage(answer: str, key_points: list[str]) -> tuple[int, int]:
    normalized_answer = normalize_text(answer)
    covered = sum(
        normalize_text(point) in normalized_answer
        for point in key_points
    )
    return covered, len(key_points)


async def run() -> int:
    args = parse_args()
    cases = load_jsonl(PROJECT_ROOT / args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    initialization_started = time.perf_counter()
    init_agentscope()
    model = (
        build_chat_model(
            thinking_type=args.thinking,
            max_tokens=1800,
            temperature=0.1,
        )
        if args.with_answers
        else None
    )
    registry = LazyAgentRegistry(model=model, cache={})
    agent = registry["rag_knowledge"]
    # Keep the context used for answer generation aligned with the evaluated K.
    agent.top_k = args.top_k
    initialization_seconds = round(
        time.perf_counter() - initialization_started,
        4,
    )

    rows = []
    expected_sources_rows: list[list[str]] = []
    retrieved_sources_rows: list[list[str]] = []

    for index, case in enumerate(cases, 1):
        started_at = time.perf_counter()
        retrieved = agent.search_knowledge(
            case["question"],
            top_k=args.top_k,
        )
        sources = [
            str(item.get("metadata", {}).get("source", ""))
            for item in retrieved
        ]
        expected_sources = case.get("expected_sources", [])
        expected_sources_rows.append(expected_sources)
        retrieved_sources_rows.append(sources)

        answer = None
        error = None
        attempt_count = 0
        retry_count = 0
        rate_limit_retry_count = 0
        timeout_retry_count = 0
        if args.with_answers:
            while attempt_count <= args.max_retries:
                attempt_count += 1
                try:
                    response = await agent.reply(
                        Msg(
                            name="rag_eval",
                            role="user",
                            content=case["question"],
                        ),
                    )
                    payload = json.loads(response.content)
                    answer = str(payload.get("answer", ""))
                    generated_error = (
                        str(payload.get("status", "")) == "error"
                        or "生成答案时出错" in answer
                        or answer.strip() == "无法生成答案"
                    )
                    if generated_error:
                        error = answer or str(payload)
                    else:
                        error = None
                        break
                except Exception as exc:
                    error = str(exc)

                reason = retry_reason(error)
                if reason is None or attempt_count > args.max_retries:
                    break
                retry_count += 1
                if reason == "rate_limit":
                    rate_limit_retry_count += 1
                if reason == "timeout":
                    timeout_retry_count += 1
                retry_delay = (
                    args.retry_base_delay
                    * (2 ** (retry_count - 1))
                )
                print(
                    f"[{index}/{len(cases)}] {case['id']} "
                    f"{reason}; retry "
                    f"{retry_count}/{args.max_retries} "
                    f"in {retry_delay:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)

        covered, total_points = key_point_coverage(
            answer or "",
            case.get("answer_key_points", []),
        )
        refused = bool(
            answer
            and any(phrase in answer for phrase in REFUSAL_PHRASES)
        )
        expected_set = set(expected_sources)
        retrieved_set = set(sources[:args.top_k])
        matching_sources = sorted(expected_set & retrieved_set)
        first_rank = next(
            (
                rank
                for rank, source in enumerate(sources, 1)
                if source in expected_set
            ),
            None,
        )
        elapsed = time.perf_counter() - started_at
        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "answerable": case.get("answerable", True),
            "expected_sources": expected_sources,
            "answer_key_points": case.get("answer_key_points", []),
            "retrieved_sources": sources,
            f"hit_at_{args.top_k}": (
                first_rank is not None if expected_sources else None
            ),
            f"all_sources_recalled_at_{args.top_k}": (
                expected_set <= retrieved_set if expected_sources else None
            ),
            f"source_recall_at_{args.top_k}": (
                round(len(matching_sources) / len(expected_set), 4)
                if expected_sources
                else None
            ),
            "matching_sources": matching_sources,
            "first_relevant_rank": first_rank,
            "answer": answer,
            "key_points_covered": covered if args.with_answers else None,
            "key_points_total": total_points if args.with_answers else None,
            "refused": refused if args.with_answers else None,
            "manual_answer_score_0_to_2": None,
            "latency_seconds": round(elapsed, 4),
            "attempt_count": attempt_count if args.with_answers else None,
            "retry_count": retry_count if args.with_answers else None,
            "rate_limit_retry_count": (
                rate_limit_retry_count if args.with_answers else None
            ),
            "timeout_retry_count": (
                timeout_retry_count if args.with_answers else None
            ),
            "error": error,
        })
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"sources={sources[:args.top_k]} {elapsed:.1f}s",
        )
        if args.delay:
            await asyncio.sleep(args.delay)

    retrieval = retrieval_metrics(
        expected_sources_rows,
        retrieved_sources_rows,
        k=args.top_k,
    )
    answerable_rows = [
        row for row in rows if row["answerable"]
    ]
    unanswerable_rows = [
        row for row in rows if not row["answerable"]
    ]
    total_points = sum(
        row["key_points_total"] or 0
        for row in answerable_rows
    )
    covered_points = sum(
        row["key_points_covered"] or 0
        for row in answerable_rows
    )
    hallucination_rate = None
    answerable_false_refusal_rate = None
    if args.with_answers and unanswerable_rows:
        hallucination_rate = round(
            sum(not row["refused"] for row in unanswerable_rows)
            / len(unanswerable_rows),
            4,
        )
    if args.with_answers and answerable_rows:
        answerable_false_refusal_rate = round(
            sum(row["refused"] for row in answerable_rows)
            / len(answerable_rows),
            4,
        )
    error_count = sum(bool(row["error"]) for row in rows)
    rate_limit_retry_count = sum(
        row["rate_limit_retry_count"] or 0 for row in rows
    )
    retry_count = sum(row["retry_count"] or 0 for row in rows)
    timeout_retry_count = sum(
        row["timeout_retry_count"] or 0 for row in rows
    )

    report = {
        "evaluation_type": "rag",
        "generated_at": utc_now_iso(),
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(
            (PROJECT_ROOT / args.dataset).read_bytes(),
        ).hexdigest(),
        "model": LLM_CONFIG["model_name"] if args.with_answers else None,
        "thinking": args.thinking if args.with_answers else None,
        "with_answers": args.with_answers,
        "sample_count": len(rows),
        "backend": getattr(agent, "backend", "unknown"),
        "initialization_seconds": initialization_seconds,
        "embedding_cache_status": getattr(
            agent,
            "embedding_cache_status",
            "not_applicable",
        ),
        "metrics": {
            **retrieval,
            "key_point_coverage": (
                round(covered_points / total_points, 4)
                if args.with_answers and total_points
                else None
            ),
            "unanswerable_hallucination_rate": hallucination_rate,
            "answerable_false_refusal_rate": answerable_false_refusal_rate,
            "error_rate": round(
                error_count / len(rows) if rows else 0.0,
                4,
            ),
            "retry_count": retry_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "timeout_retry_count": timeout_retry_count,
            "manual_answer_score_mean": None,
        },
        "rows": rows,
        "caveats": [
            "Manual answer scores must be reviewed before resume use.",
            "Substring key-point coverage is diagnostic, not semantic accuracy.",
            "Retrieval and answer-generation metrics must be reported separately.",
        ],
    }
    destination = write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
