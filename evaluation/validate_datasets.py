"""Validate evaluation dataset schemas and label/source consistency."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_jsonl


INTENTS = {
    "itinerary_planning",
    "memory_query",
    "preference",
    "rag_knowledge",
    "information_query",
    "event_collection",
}


def validate_intents() -> int:
    rows = load_jsonl(
        PROJECT_ROOT
        / "evaluation/datasets/intent_eval.sample.jsonl",
    )
    for row in rows:
        expected = set(row.get("expected_intents", []))
        priorities = set(row.get("expected_priorities", {}))
        unknown = (expected | priorities) - INTENTS
        assert not unknown, f"{row['id']} has unknown labels: {unknown}"
        assert expected == priorities, (
            f"{row['id']} intent/schedule labels differ: "
            f"{expected} vs {priorities}"
        )
        assert isinstance(row.get("history", []), list)
        assert row.get("query", "").strip()
    return len(rows)


def validate_rag() -> int:
    rows = load_jsonl(
        PROJECT_ROOT / "evaluation/datasets/rag_eval.sample.jsonl",
    )
    document_dir = (
        PROJECT_ROOT
        / ".claude/skills/ask-question/data/documents"
    )
    available_sources = {
        path.name for path in document_dir.glob("*.txt")
    }
    for row in rows:
        sources = set(row.get("expected_sources", []))
        unknown = sources - available_sources
        assert not unknown, f"{row['id']} unknown sources: {unknown}"
        if row.get("answerable", True):
            assert sources, f"{row['id']} answerable but has no source"
            assert row.get("answer_key_points"), (
                f"{row['id']} answerable but has no key points"
            )
        else:
            assert not sources, (
                f"{row['id']} unanswerable but has expected sources"
            )
    return len(rows)


def validate_latency() -> int:
    rows = load_jsonl(
        PROJECT_ROOT
        / "evaluation/datasets/latency_queries.sample.jsonl",
    )
    for row in rows:
        assert row.get("query", "").strip()
        assert row.get("category", "").strip()
    return len(rows)


def main():
    intent_count = validate_intents()
    rag_count = validate_rag()
    latency_count = validate_latency()
    print(f"PASS: {intent_count} intent rows")
    print(f"PASS: {rag_count} RAG rows")
    print(f"PASS: {latency_count} latency rows")


if __name__ == "__main__":
    main()
