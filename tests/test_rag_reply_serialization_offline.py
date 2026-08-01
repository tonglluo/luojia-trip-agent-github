"""Offline regression for RAG reply JSON serialization scope."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
from pathlib import Path

from agentscope.message import Msg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = (
    PROJECT_ROOT
    / ".claude"
    / "skills"
    / "ask-question"
    / "script"
    / "agent.py"
)


def load_agent_class():
    spec = importlib.util.spec_from_file_location(
        "rag_agent_serialization_test",
        AGENT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RAG agent: {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent


async def main() -> None:
    agent_class = load_agent_class()
    agent = object.__new__(agent_class)
    agent.name = "RAGSerializationTest"
    agent.initialized = False

    raw_reply = inspect.unwrap(agent_class.reply)
    response = await raw_reply(
        agent,
        Msg(name="user", role="user", content="测试"),
    )
    payload = json.loads(response.content)
    assert payload["status"] == "error"
    assert "message" in payload
    print("PASS: RAG reply serializes JSON without local-scope errors")


if __name__ == "__main__":
    asyncio.run(main())
