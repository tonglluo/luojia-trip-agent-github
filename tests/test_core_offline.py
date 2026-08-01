"""Offline smoke tests for core orchestration and memory behavior.

Run with:
    .venv/Scripts/python.exe tests/test_core_offline.py

These tests do not call an LLM, the internet, Milvus, or Hugging Face.
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.message import Msg

from agents.orchestration_agent import OrchestrationAgent
from context.memory_manager import MemoryManager


class ConcurrencyTracker:
    def __init__(self):
        self.active = 0
        self.max_active = 0


class FakeAgent:
    """Small async agent used to verify orchestration without an API key."""

    def __init__(self, name, tracker=None, received_inputs=None):
        self.name = name
        self.tracker = tracker
        self.received_inputs = received_inputs if received_inputs is not None else {}

    async def reply(self, msg):
        payload = json.loads(msg.content)
        self.received_inputs[self.name] = payload

        if self.tracker:
            self.tracker.active += 1
            self.tracker.max_active = max(
                self.tracker.max_active,
                self.tracker.active,
            )
            await asyncio.sleep(0.02)
            self.tracker.active -= 1

        return Msg(
            name=self.name,
            role="assistant",
            content=json.dumps({"source": self.name}, ensure_ascii=False),
        )


class PreferenceSnapshotReader:
    def __init__(self, memory_manager, observations):
        self.name = "memory_query"
        self.memory_manager = memory_manager
        self.observations = observations

    async def reply(self, msg):
        value = self.memory_manager.long_term.get_preference("airline")
        self.observations.append(value)
        await asyncio.sleep(0.02)
        return Msg(
            name=self.name,
            role="assistant",
            content=json.dumps({"observed": value}, ensure_ascii=False),
        )


class PreferenceUpdateProducer:
    def __init__(self):
        self.name = "preference"

    async def reply(self, msg):
        return Msg(
            name=self.name,
            role="assistant",
            content=json.dumps({
                "preferences": [
                    {
                        "type": "airline",
                        "value": "东航",
                        "action": "replace",
                    },
                ],
            }, ensure_ascii=False),
        )


def test_memory_persistence():
    with tempfile.TemporaryDirectory() as storage_path:
        first_session = MemoryManager(
            user_id="offline_user",
            session_id="session_1",
            storage_path=storage_path,
        )
        first_session.add_message("user", "我喜欢亚朵")
        first_session.long_term.save_preference("hotel_brands", ["亚朵"])
        assert len(first_session.short_term.messages) == 1

        first_session.end_session()
        assert first_session.short_term.messages == []

        second_session = MemoryManager(
            user_id="offline_user",
            session_id="session_2",
            storage_path=storage_path,
        )
        assert second_session.short_term.messages == []
        assert second_session.long_term.get_preference("hotel_brands") == [
            "亚朵",
        ]
        assert len(second_session.long_term.get_chat_history()) == 1


async def test_priority_orchestration():
    tracker = ConcurrencyTracker()
    received_inputs = {}
    registry = {
        "memory_query": FakeAgent(
            "memory_query",
            tracker,
            received_inputs,
        ),
        "event_collection": FakeAgent(
            "event_collection",
            tracker,
            received_inputs,
        ),
        "itinerary_planning": FakeAgent(
            "itinerary_planning",
            received_inputs=received_inputs,
        ),
    }
    orchestrator = OrchestrationAgent(agent_registry=registry)
    decision = {
        "intents": [{"type": "itinerary_planning"}],
        "key_entities": {"destination": "北京"},
        "rewritten_query": "根据历史偏好规划北京行程",
        "agent_schedule": [
            {"agent_name": "memory_query", "priority": 1},
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ],
    }

    response = await orchestrator.reply(
        Msg(
            name="IntentionAgent",
            role="assistant",
            content=json.dumps(decision, ensure_ascii=False),
        ),
    )
    result = json.loads(response.content)

    assert result["status"] == "completed"
    assert result["agents_executed"] == 3
    assert tracker.max_active == 2

    planning_input = received_inputs["itinerary_planning"]
    assert len(planning_input["previous_results"]) == 2
    previous_names = {
        item["agent_name"]
        for item in planning_input["previous_results"]
    }
    assert previous_names == {"memory_query", "event_collection"}


async def test_parallel_memory_read_before_preference_commit():
    with tempfile.TemporaryDirectory() as storage_path:
        memory_manager = MemoryManager(
            user_id="snapshot_user",
            session_id="snapshot_session",
            storage_path=storage_path,
        )
        memory_manager.long_term.save_preference("airline", "南航")
        observations = []
        orchestrator = OrchestrationAgent(
            agent_registry={
                "memory_query": PreferenceSnapshotReader(
                    memory_manager,
                    observations,
                ),
                "preference": PreferenceUpdateProducer(),
            },
            memory_manager=memory_manager,
        )
        decision = {
            "intents": [
                {"type": "memory_query"},
                {"type": "preference"},
            ],
            "rewritten_query": "告诉我原来的航空公司偏好，再改成东航",
            "agent_schedule": [
                {"agent_name": "memory_query", "priority": 1},
                {"agent_name": "preference", "priority": 1},
            ],
        }
        await orchestrator.reply(
            Msg(
                name="IntentionAgent",
                role="assistant",
                content=json.dumps(decision, ensure_ascii=False),
            ),
        )
        assert observations == ["南航"]
        assert (
            memory_manager.long_term.get_preference("airline")
            == "东航"
        )


async def main():
    test_memory_persistence()
    await test_priority_orchestration()
    await test_parallel_memory_read_before_preference_commit()
    print("PASS: memory persistence")
    print("PASS: same-priority agents run concurrently")
    print("PASS: priority-2 agent receives priority-1 results")
    print("PASS: parallel memory reads old snapshot before preference commit")


if __name__ == "__main__":
    asyncio.run(main())
