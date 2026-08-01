"""Offline tests for RAG evaluation retry classification."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_rag_eval import retry_reason


def main() -> None:
    assert retry_reason("Error code: 429, code 1302") == "rate_limit"
    assert retry_reason("Request timed out.") == "timeout"
    assert retry_reason("HTTP 503 service unavailable") == "server_error"
    assert retry_reason("Connection reset by peer") == "connection_error"
    assert retry_reason("invalid JSON response") is None
    print("PASS: RAG retry classification")


if __name__ == "__main__":
    main()
