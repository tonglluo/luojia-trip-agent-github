"""Offline tests for deterministic intent-routing invariants."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.intention_agent import IntentionAgent


def _intent_types(result: dict) -> set[str]:
    return {item["type"] for item in result["intents"]}


def _scheduled_agents(result: dict) -> dict[str, int]:
    return {
        item["agent_name"]: item["priority"]
        for item in result["agent_schedule"]
    }


def test_itinerary_dependency_and_spurious_preference() -> None:
    raw = {
        "intents": [
            {"type": "itinerary_planning"},
            {"type": "preference"},
        ],
        "agent_schedule": [
            {"agent_name": "preference", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "帮我规划一个北京三日游",
    )
    assert _intent_types(result) == {
        "event_collection",
        "itinerary_planning",
    }
    assert _scheduled_agents(result) == {
        "event_collection": 1,
        "itinerary_planning": 2,
    }


def test_explicit_preference_is_preserved() -> None:
    raw = {
        "intents": [
            {"type": "preference"},
            {"type": "itinerary_planning"},
        ],
        "agent_schedule": [
            {"agent_name": "preference", "priority": 9},
            {"agent_name": "itinerary_planning", "priority": 9},
        ],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "我喜欢亚朵，下周从上海去北京出差三天，帮我规划",
    )
    assert _intent_types(result) == {
        "preference",
        "event_collection",
        "itinerary_planning",
    }
    assert _scheduled_agents(result) == {
        "preference": 1,
        "event_collection": 1,
        "itinerary_planning": 2,
    }


def test_unknown_intent_and_orphan_schedule_are_removed() -> None:
    raw = {
        "intents": [
            {"type": "greeting"},
            {"type": "event_collection"},
        ],
        "agent_schedule": [
            {"agent_name": "preference", "priority": 1},
            {"agent_name": "event_collection", "priority": 1},
        ],
    }
    result = IntentionAgent._normalize_result(raw, "你好，很高兴认识你")
    assert result["intents"] == []
    assert result["agent_schedule"] == []


def test_public_policy_does_not_trigger_enterprise_rag() -> None:
    raw = {
        "intents": [
            {"type": "information_query"},
            {"type": "rag_knowledge"},
        ],
        "agent_schedule": [
            {"agent_name": "information_query", "priority": 1},
            {"agent_name": "rag_knowledge", "priority": 1},
        ],
    }
    result = IntentionAgent._normalize_result(raw, "上海今天限行吗？")
    assert _intent_types(result) == {"information_query"}
    assert _scheduled_agents(result) == {"information_query": 1}


def test_non_travel_information_is_out_of_scope() -> None:
    raw = {
        "intents": [{"type": "information_query"}],
        "key_entities": {
            "origin": None,
            "destination": None,
            "other": "费马大定理",
        },
        "agent_schedule": [
            {"agent_name": "information_query", "priority": 1},
        ],
    }
    result = IntentionAgent._normalize_result(raw, "帮我证明费马大定理")
    assert result["intents"] == []
    assert result["agent_schedule"] == []


def test_elliptical_context_preference_is_preserved() -> None:
    raw = {
        "intents": [{"type": "preference"}],
        "agent_schedule": [{"agent_name": "preference", "priority": 1}],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "不要临街的房间",
        "用户: 我想更新住宿习惯\n助手: 你希望记录什么住宿偏好？",
    )
    assert _intent_types(result) == {"preference"}

    result = IntentionAgent._normalize_result(
        raw,
        "以后都要远离电梯的房间",
        "用户: 我想补充一个长期住宿偏好\n助手: 请告诉我具体要求。",
    )
    assert _intent_types(result) == {"preference"}


def test_persistent_preference_without_first_person_is_preserved() -> None:
    queries = (
        "以后出差优先坐高铁",
        "住宿预算习惯控制在每晚五百以内",
        "订酒店时优先选择无烟房，这个以后都按这个来",
        "住宿费用每晚最高七百元，帮我保存这个标准",
        "以后酒店必须有洗衣房",
    )
    for query in queries:
        raw = {
            "intents": [{"type": "preference"}],
            "agent_schedule": [
                {"agent_name": "preference", "priority": 1},
            ],
        }
        result = IntentionAgent._normalize_result(raw, query)
        assert _intent_types(result) == {"preference"}


def test_existing_preference_question_does_not_write_preference() -> None:
    raw = {
        "intents": [
            {"type": "memory_query"},
            {"type": "information_query"},
            {"type": "preference"},
        ],
        "key_entities": {"destination": "北京"},
        "agent_schedule": [
            {"agent_name": "memory_query", "priority": 1},
            {"agent_name": "information_query", "priority": 1},
            {"agent_name": "preference", "priority": 1},
        ],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "我喜欢的酒店在北京有门店吗？",
    )
    assert _intent_types(result) == {
        "memory_query",
        "information_query",
    }


def test_budget_and_cancellation_preference_mutations() -> None:
    queries = (
        "酒店预算每晚不要超过六百元",
        "取消我之前保存的靠过道座位偏好",
    )
    for query in queries:
        raw = {
            "intents": [{"type": "preference"}],
            "agent_schedule": [
                {"agent_name": "preference", "priority": 1},
            ],
        }
        result = IntentionAgent._normalize_result(raw, query)
        assert _intent_types(result) == {"preference"}


def test_explicit_budget_preference_can_be_recovered() -> None:
    raw = {
        "intents": [],
        "agent_schedule": [],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "酒店预算每晚不要超过六百元",
    )
    assert _intent_types(result) == {"preference"}
    assert _scheduled_agents(result) == {"preference": 1}


def test_generic_delete_does_not_recover_preference() -> None:
    raw = {
        "intents": [],
        "agent_schedule": [],
    }
    result = IntentionAgent._normalize_result(
        raw,
        "删除这份行程文件",
    )
    assert _intent_types(result) == set()
    assert _scheduled_agents(result) == {}


def main() -> None:
    test_itinerary_dependency_and_spurious_preference()
    test_explicit_preference_is_preserved()
    test_unknown_intent_and_orphan_schedule_are_removed()
    test_public_policy_does_not_trigger_enterprise_rag()
    test_non_travel_information_is_out_of_scope()
    test_elliptical_context_preference_is_preserved()
    test_persistent_preference_without_first_person_is_preserved()
    test_existing_preference_question_does_not_write_preference()
    test_budget_and_cancellation_preference_mutations()
    test_explicit_budget_preference_can_be_recovered()
    test_generic_delete_does_not_recover_preference()
    print("PASS: itinerary dependency")
    print("PASS: explicit preference")
    print("PASS: supported-intent schedule consistency")
    print("PASS: public information versus enterprise RAG boundary")
    print("PASS: non-travel information boundary")
    print("PASS: contextual and persistent preference boundaries")
    print("PASS: budget and cancellation preference mutations")
    print("PASS: high-precision preference recovery")


if __name__ == "__main__":
    main()
