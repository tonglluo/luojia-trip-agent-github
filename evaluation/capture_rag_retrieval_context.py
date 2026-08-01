"""Capture the exact Top-K chunks used to audit a saved RAG answer run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.lazy_agent_registry import LazyAgentRegistry
from config_agentscope import init_agentscope
from evaluation.common import utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        default="evaluation/reports/rag_formal_v1_answers_attempt3.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_v1_retrieval_context.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    answer_path = PROJECT_ROOT / args.answers
    answer_report = json.loads(answer_path.read_text(encoding="utf-8"))
    if answer_report["metrics"]["error_rate"] != 0:
        raise SystemExit("Answer report must be error-free")

    initialization_started = time.perf_counter()
    init_agentscope()
    registry = LazyAgentRegistry(model=None, cache={})
    agent = registry["rag_knowledge"]
    agent.top_k = args.top_k
    initialization_seconds = round(
        time.perf_counter() - initialization_started,
        4,
    )

    rows: list[dict[str, Any]] = []
    mismatch_rows = []
    for index, answer_row in enumerate(answer_report["rows"], 1):
        retrieved = agent.search_knowledge(
            answer_row["question"],
            top_k=args.top_k,
        )
        documents = [
            {
                "rank": rank,
                "id": item.get("id"),
                "source": str(
                    item.get("metadata", {}).get("source", ""),
                ),
                "chunk_index": item.get(
                    "metadata",
                    {},
                ).get("chunk_index"),
                "content": item.get("content", ""),
                "score": item.get("distance"),
            }
            for rank, item in enumerate(retrieved, 1)
        ]
        actual_sources = [item["source"] for item in documents]
        expected_sources = list(answer_row["retrieved_sources"])
        sources_match = actual_sources == expected_sources
        if not sources_match:
            mismatch_rows.append({
                "id": answer_row["id"],
                "saved_sources": expected_sources,
                "reconstructed_sources": actual_sources,
            })
        rows.append({
            "id": answer_row["id"],
            "question": answer_row["question"],
            "answer": answer_row["answer"],
            "answer_key_points": answer_row["answer_key_points"],
            "saved_retrieved_sources": expected_sources,
            "reconstructed_sources": actual_sources,
            "sources_match_saved_run": sources_match,
            "documents": documents,
        })
        print(
            f"[{index}/{len(answer_report['rows'])}] "
            f"{answer_row['id']} match={sources_match}",
        )

    result = {
        "generated_at": utc_now_iso(),
        "source_answer_report": args.answers,
        "source_answer_report_sha256": sha256(answer_path),
        "dataset": answer_report["dataset"],
        "dataset_sha256": answer_report["dataset_sha256"],
        "sample_count": len(rows),
        "top_k": args.top_k,
        "backend": getattr(agent, "backend", "unknown"),
        "embedding_cache_status": getattr(
            agent,
            "embedding_cache_status",
            "not_applicable",
        ),
        "initialization_seconds": initialization_seconds,
        "source_sequence_match_count": sum(
            row["sources_match_saved_run"] for row in rows
        ),
        "source_sequence_mismatch_count": len(mismatch_rows),
        "source_sequence_mismatches": mismatch_rows,
        "rows": rows,
        "note": (
            "Chunk reconstruction is accepted as exact evidence only when "
            "the ordered source sequence matches the saved answer run. "
            "Embedding/document changes require a new capture."
        ),
    }
    destination = write_json(PROJECT_ROOT / args.output, result)
    print(json.dumps({
        "sample_count": result["sample_count"],
        "source_sequence_match_count": result[
            "source_sequence_match_count"
        ],
        "source_sequence_mismatch_count": result[
            "source_sequence_mismatch_count"
        ],
        "initialization_seconds": initialization_seconds,
        "embedding_cache_status": result["embedding_cache_status"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return 0 if not mismatch_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
