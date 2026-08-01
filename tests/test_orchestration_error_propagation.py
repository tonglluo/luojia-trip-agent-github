"""Offline regression test for nested agent error propagation."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.agent import AgentBase
from agentscope.message import Msg

from agents.orchestration_agent import OrchestrationAgent


class StatusErrorAgent(AgentBase):
    async def reply(self, x=None) -> Msg:
        return Msg(
            name="status_error",
            role="assistant",
            content=json.dumps({
                "status": "error",
                "message": "simulated timeout",
            }),
        )


class RateLimitedOnceAgent(AgentBase):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def reply(self, x=None) -> Msg:
        self.calls += 1
        if self.calls == 1:
            content = {
                "status": "error",
                "error": "Error code: 429, code 1302 rate limit",
            }
        else:
            content = {"status": "success", "answer": "ok"}
        return Msg(
            name="rate_limited_once",
            role="assistant",
            content=json.dumps(content),
        )


async def run_case() -> None:
    orchestrator = OrchestrationAgent(
        agent_registry={"memory_query": StatusErrorAgent()},
    )
    response = await orchestrator.reply(Msg(
        name="intent",
        role="assistant",
        content=json.dumps({
            "intents": [{"type": "memory_query"}],
            "agent_schedule": [{
                "agent_name": "memory_query",
                "priority": 1,
                "reason": "test",
                "expected_output": "test",
            }],
            "rewritten_query": "test",
        }),
    ))
    payload = json.loads(response.content)
    assert payload["status"] == "partial_failure"
    assert payload["errors"] == 1
    assert payload["results"][0]["status"] == "error"

    flaky_agent = RateLimitedOnceAgent()
    retrying_orchestrator = OrchestrationAgent(
        agent_registry={"rag_knowledge": flaky_agent},
    )
    retry_response = await retrying_orchestrator.reply(Msg(
        name="intent",
        role="assistant",
        content=json.dumps({
            "intents": [{"type": "rag_knowledge"}],
            "agent_schedule": [{
                "agent_name": "rag_knowledge",
                "priority": 1,
                "reason": "test retry",
                "expected_output": "test",
            }],
            "rewritten_query": "test",
        }),
    ))
    retry_payload = json.loads(retry_response.content)
    assert retry_payload["status"] == "completed"
    assert retry_payload["agent_retry_count"] == 1
    assert retry_payload["results"][0]["retry_count"] == 1
    assert flaky_agent.calls == 2


if __name__ == "__main__":
    asyncio.run(run_case())
    print("PASS: nested errors propagate and retriable 429 errors retry")
