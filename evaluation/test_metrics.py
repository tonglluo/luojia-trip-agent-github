"""Offline unit tests for evaluation calculations."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    entity_field_score,
    latency_summary,
    multilabel_classification,
    retrieval_metrics,
)


def main():
    classification = multilabel_classification(
        [{"a"}, {"a", "b"}],
        [{"a"}, {"b"}],
        ["a", "b"],
    )
    assert classification["exact_match"] == 0.5
    assert classification["per_label"]["b"]["recall"] == 1.0

    retrieval = retrieval_metrics(
        [["doc1"], ["doc2"]],
        [["doc1", "doc3"], ["doc9", "doc2"]],
        k=2,
    )
    assert retrieval["recall_at_2"] == 1.0
    assert retrieval["mrr"] == 0.75

    multi_source_retrieval = retrieval_metrics(
        [["doc1", "doc2"], ["doc3"]],
        [["doc1", "doc9"], ["doc3", "doc8"]],
        k=2,
    )
    assert multi_source_retrieval["any_source_hit_at_2"] == 1.0
    assert multi_source_retrieval["all_sources_recalled_at_2"] == 0.5
    assert multi_source_retrieval["mean_source_recall_at_2"] == 0.75

    correct, total, checks = entity_field_score(
        {"origin": "上海", "brands": ["亚朵", "汉庭"]},
        {"origin": " 上海 ", "brands": ["汉庭", "亚朵"]},
    )
    assert (correct, total) == (2, 2)

    correct, total, _ = entity_field_score(
        {
            "other": {
                "contains_all": ["无烟房", "酒店"],
            },
        },
        {
            "other": {
                "preference_type": "酒店房型",
                "preference_value": "优先选择无烟房",
            },
        },
    )
    assert (correct, total) == (1, 1)

    negative_preference = {
        "other": {
            "contains_all_groups": [
                ["不喜欢", "避免", "不坐"],
                ["红眼航班"],
            ],
        },
    }
    correct, total, _ = entity_field_score(
        negative_preference,
        {
            "other": "用户希望避免红眼航班",
        },
    )
    assert (correct, total) == (1, 1)

    correct, total, _ = entity_field_score(
        negative_preference,
        {
            "other": "用户偏好红眼航班",
        },
    )
    assert (correct, total) == (0, 1)

    budget_preference = {
        "other": {
            "contains_all_groups": [
                ["每晚", "每夜"],
                ["五百", "500"],
                ["以内", "上限", "不超过"],
            ],
        },
    }
    correct, total, _ = entity_field_score(
        budget_preference,
        {"other": "住宿预算上限：每晚500元"},
    )
    assert (correct, total) == (1, 1)

    correct, total, _ = entity_field_score(
        budget_preference,
        {"other": "住宿预算为500元"},
    )
    assert (correct, total) == (0, 1)

    deletion_preference = {
        "other": {
            "contains_all_groups": [
                ["删除", "取消", "不再偏好", "移除"],
                ["经济型酒店"],
            ],
        },
    }
    correct, total, _ = entity_field_score(
        deletion_preference,
        {"other": "取消经济型酒店偏好"},
    )
    assert (correct, total) == (1, 1)

    correct, total, _ = entity_field_score(
        deletion_preference,
        {"other": "偏好经济型酒店"},
    )
    assert (correct, total) == (0, 1)

    correct, total, _ = entity_field_score(
        {"duration": {"any_of": ["三天", "3天", "3日"]}},
        {"duration": "3天"},
    )
    assert (correct, total) == (1, 1)

    correct, total, _ = entity_field_score(
        {"origin": "成都", "destination": "北京"},
        {
            "origin": "成都（来自上下文）",
            "destination": "北京（来自上下文）",
        },
    )
    assert (correct, total) == (2, 2)

    latency = latency_summary([
        {
            "status": "success",
            "total_seconds": 1.0,
            "intent_seconds": 0.3,
            "execution_seconds": 0.7,
        },
        {
            "status": "success",
            "total_seconds": 3.0,
            "intent_seconds": 1.0,
            "execution_seconds": 2.0,
        },
        {
            "status": "failure",
            "total_seconds": 10.0,
            "intent_seconds": 0.0,
            "execution_seconds": 0.0,
        },
    ])
    assert latency["success_rate"] == 0.6667
    assert latency["total_seconds"]["p50"] == 2.0

    print("PASS: intent metrics")
    print("PASS: retrieval metrics")
    print("PASS: entity metrics")
    print("PASS: latency metrics")


if __name__ == "__main__":
    main()
