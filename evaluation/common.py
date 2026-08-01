"""Shared helpers for reproducible evaluation scripts."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"

if sys.platform.startswith("win"):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL and enforce non-empty, unique IDs."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number} is not valid JSON: {exc}",
                ) from exc

            row_id = str(row.get("id", "")).strip()
            if not row_id:
                raise ValueError(f"{source}:{line_number} has no id")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate id in {source}: {row_id}")
            seen_ids.add(row_id)
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset is empty: {source}")
    return rows


def write_json(path: str | Path, data: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Linear percentile compatible with small benchmark samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def build_chat_model(
    *,
    thinking_type: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
):
    """Build the same AgentScope model used by the production CLI."""
    from agentscope.model import OpenAIChatModel
    from config import LLM_CONFIG, SYSTEM_CONFIG, validate_llm_config

    validate_llm_config()
    resolved_thinking = thinking_type or LLM_CONFIG["thinking_type"]
    if resolved_thinking not in {"enabled", "disabled"}:
        raise ValueError("thinking_type must be enabled or disabled")

    return OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={
            "base_url": LLM_CONFIG["base_url"],
            "timeout": float(SYSTEM_CONFIG.get("timeout", 60)),
            # Evaluation runners own retry/backoff so every retry is visible
            # in raw reports. Disable the SDK's additional hidden retries.
            "max_retries": 0,
        },
        generate_kwargs={
            "temperature": (
                LLM_CONFIG["temperature"]
                if temperature is None
                else temperature
            ),
            "max_tokens": (
                LLM_CONFIG["max_tokens"]
                if max_tokens is None
                else max_tokens
            ),
            "extra_body": {
                "thinking": {"type": resolved_thinking},
            },
        },
    )
