"""Audit an intent dataset before it is frozen or evaluated."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_jsonl, utc_now_iso, write_json
from agents.intention_agent import IntentionAgent


VALID_INTENTS = {
    "itinerary_planning",
    "memory_query",
    "preference",
    "rag_knowledge",
    "information_query",
    "event_collection",
}
EXPECTED_PRIORITIES = {
    "memory_query": 1,
    "event_collection": 1,
    "preference": 1,
    "information_query": 1,
    "rag_knowledge": 1,
    "itinerary_planning": 2,
}
INTENTS_REQUIRING_OTHER = {
    "memory_query",
    "preference",
    "rag_knowledge",
    "information_query",
}
OTHER_MATCHERS = {
    "any_of",
    "contains_all",
    "contains_any",
    "contains_all_groups",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/intent_eval.holdout.candidate.jsonl",
    )
    parser.add_argument(
        "--development-set",
        default="evaluation/datasets/intent_eval.sample.jsonl",
    )
    parser.add_argument(
        "--comparison-set",
        action="append",
        default=[],
        help=(
            "Additional JSONL set checked for exact normalized-query "
            "overlap. May be provided more than once."
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_holdout_candidate_audit.json",
    )
    parser.add_argument(
        "--review-output",
        default="evaluation/reports/intent_holdout_review.md",
    )
    parser.add_argument("--minimum-label-count", type=int, default=10)
    return parser.parse_args()


def _normalize_query(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _representative_entities(entities: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for field, value in entities.items():
        if isinstance(value, dict) and value.get("any_of"):
            result[field] = value["any_of"][0]
        else:
            result[field] = value
    return result


def _write_review(
    path: Path,
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "# 意图识别独立测试集人工复核清单",
        "",
        f"- 状态：候选集，尚未冻结",
        f"- 样本数：{len(rows)}",
        f"- SHA-256：`{audit['dataset_sha256']}`",
        f"- 自动审计问题数：{audit['issue_count']}",
        "",
        "复核规则：逐条确认 Query、上下文、意图集合、调度优先级和实体答案。"
        "如需修改，必须在首次模型评测前完成。",
        "",
        "|确认|ID|场景|上文|Query|期望意图|优先级|期望实体|标注说明|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        query = str(row["query"]).replace("|", "\\|")
        history = "<br>".join(
            f"{item.get('role', '')}: {item.get('content', '')}"
            for item in row.get("history", [])
        ).replace("|", "\\|") or "—"
        intents = ", ".join(row.get("expected_intents", [])) or "空"
        priorities = json.dumps(
            row.get("expected_priorities", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        entities = json.dumps(
            row.get("expected_entities", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        annotation_note = str(
            row.get("annotation_note", ""),
        ).replace("|", "\\|") or "—"
        lines.append(
            f"|[ ]|{row['id']}|{row.get('category', '')}|{history}|"
            f"{query}|{intents}|`{priorities}`|`{entities}`|"
            f"{annotation_note}|",
        )
    lines.extend([
        "",
        "## 冻结前签字",
        "",
        "- [ ] 60 条 Query 均已人工检查",
        "- [ ] 多意图样本的意图集合无遗漏",
        "- [ ] 调度优先级与产品规则一致",
        "- [ ] 实体字段只标注可以客观判断的答案",
        "- [ ] 对争议样本已有书面裁决",
        "- 复核人：________________",
        "- 日期：__________________",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(
    rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    dataset_path: Path,
    minimum_label_count: int,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    ids = [str(row.get("id", "")) for row in rows]
    normalized_queries = [
        _normalize_query(str(row.get("query", "")))
        for row in rows
    ]
    development_queries = {
        _normalize_query(str(row.get("query", "")))
        for row in development_rows
    }

    duplicate_ids = sorted(
        item for item, count in Counter(ids).items() if count > 1
    )
    duplicate_queries = sorted(
        item
        for item, count in Counter(normalized_queries).items()
        if count > 1
    )
    overlap_queries = sorted(
        set(normalized_queries) & development_queries,
    )
    for duplicate in duplicate_ids:
        issues.append({
            "scope": duplicate,
            "code": "duplicate_id",
            "message": "ID 在候选集中重复。",
        })
    for duplicate in duplicate_queries:
        issues.append({
            "scope": duplicate,
            "code": "duplicate_query",
            "message": "归一化后的 Query 在候选集中重复。",
        })
    for overlap in overlap_queries:
        issues.append({
            "scope": overlap,
            "code": "development_overlap",
            "message": "候选 Query 与开发集完全重复。",
        })

    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    multi_intent_count = 0
    context_count = 0
    unsupported_count = 0
    entity_annotated_count = 0

    for index, row in enumerate(rows, 1):
        row_id = str(row.get("id", f"row_{index}"))
        intents = list(row.get("expected_intents", []))
        intent_set = set(intents)
        priorities = row.get("expected_priorities", {})
        entities = row.get("expected_entities", {})
        category = str(row.get("category", ""))

        label_counts.update(intent_set)
        category_counts.update([category])
        multi_intent_count += len(intent_set) > 1
        context_count += bool(row.get("history"))
        unsupported_count += not intent_set
        entity_annotated_count += bool(entities)

        unknown = intent_set - VALID_INTENTS
        if unknown:
            issues.append({
                "scope": row_id,
                "code": "unknown_intent",
                "message": f"未知意图：{sorted(unknown)}",
            })
        if len(intents) != len(intent_set):
            issues.append({
                "scope": row_id,
                "code": "duplicate_intent",
                "message": "同一条样本包含重复意图。",
            })

        expected_priority_map = {
            intent: EXPECTED_PRIORITIES[intent]
            for intent in intent_set
            if intent in EXPECTED_PRIORITIES
        }
        if priorities != expected_priority_map:
            issues.append({
                "scope": row_id,
                "code": "priority_mismatch",
                "message": (
                    f"优先级应为 {expected_priority_map}，实际为 {priorities}。"
                ),
            })

        if (
            "itinerary_planning" in intent_set
            and "event_collection" not in intent_set
        ):
            issues.append({
                "scope": row_id,
                "code": "missing_event_collection",
                "message": "行程规划样本必须同时标注 event_collection。",
            })
        if (
            "event_collection" in intent_set
            and "itinerary_planning" not in intent_set
        ):
            issues.append({
                "scope": row_id,
                "code": "orphan_event_collection",
                "message": "当前产品口径下 event_collection 仅用于行程规划。",
            })
        if "itinerary_planning" in intent_set and not entities.get(
            "destination",
        ):
            issues.append({
                "scope": row_id,
                "code": "missing_destination_annotation",
                "message": "行程规划样本缺少 destination 标注。",
            })
        if not row.get("query"):
            issues.append({
                "scope": row_id,
                "code": "empty_query",
                "message": "Query 不能为空。",
            })

        other = entities.get("other")
        if intent_set & INTENTS_REQUIRING_OTHER and not other:
            issues.append({
                "scope": row_id,
                "code": "missing_other_annotation",
                "message": (
                    "memory/preference/RAG/information 样本必须用 other "
                    "标注完整的非时空语义。"
                ),
            })
        if other is not None:
            if not isinstance(other, dict):
                issues.append({
                    "scope": row_id,
                    "code": "invalid_other_matcher",
                    "message": "other 必须是包含一个匹配算子的对象。",
                })
            else:
                operators = set(other) & OTHER_MATCHERS
                if len(operators) != 1 or set(other) != operators:
                    issues.append({
                        "scope": row_id,
                        "code": "invalid_other_matcher",
                        "message": (
                            "other 必须且只能使用一个受支持的匹配算子："
                            f"{sorted(OTHER_MATCHERS)}。"
                        ),
                    })
                elif "contains_all_groups" in other:
                    groups = other["contains_all_groups"]
                    valid_groups = (
                        isinstance(groups, list)
                        and bool(groups)
                        and all(
                            isinstance(group, list)
                            and bool(group)
                            and all(
                                isinstance(item, str) and bool(item.strip())
                                for item in group
                            )
                            for group in groups
                        )
                    )
                    if not valid_groups:
                        issues.append({
                            "scope": row_id,
                            "code": "invalid_other_groups",
                            "message": (
                                "contains_all_groups 必须是非空二维字符串数组，"
                                "且每组至少包含一个非空候选表达。"
                            ),
                        })
        if (
            "preference" in intent_set
            and isinstance(other, dict)
            and "contains_all_groups" not in other
        ):
            issues.append({
                "scope": row_id,
                "code": "incomplete_preference_semantics",
                "message": (
                    "偏好样本必须使用 contains_all_groups，同时验证偏好内容"
                    "及极性、增删或更新动作。"
                ),
            })

        ideal_result = {
            "intents": [
                {"type": intent}
                for intent in intents
            ],
            "key_entities": _representative_entities(entities),
            "agent_schedule": [
                {"agent_name": name, "priority": priority}
                for name, priority in priorities.items()
            ],
        }
        context = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')}"
            for item in row.get("history", [])
        )
        normalized = IntentionAgent._normalize_result(
            ideal_result,
            str(row.get("query", "")),
            context,
        )
        normalized_intents = {
            item.get("type")
            for item in normalized.get("intents", [])
        }
        normalized_priorities = {
            item.get("agent_name"): item.get("priority")
            for item in normalized.get("agent_schedule", [])
        }
        if normalized_intents != intent_set:
            issues.append({
                "scope": row_id,
                "code": "postprocessing_intent_conflict",
                "message": (
                    "确定性后处理会改变人工标注意图："
                    f"{sorted(intent_set)} -> {sorted(normalized_intents)}"
                ),
            })
        if normalized_priorities != priorities:
            issues.append({
                "scope": row_id,
                "code": "postprocessing_schedule_conflict",
                "message": (
                    "确定性后处理会改变人工标注调度："
                    f"{priorities} -> {normalized_priorities}"
                ),
            })

    for label in sorted(VALID_INTENTS):
        if label_counts[label] < minimum_label_count:
            issues.append({
                "scope": label,
                "code": "insufficient_label_coverage",
                "message": (
                    f"正例数 {label_counts[label]} 少于要求 "
                    f"{minimum_label_count}。"
                ),
            })

    return {
        "generated_at": utc_now_iso(),
        "status": "pass" if not issues else "needs_review",
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "sample_count": len(rows),
        "development_sample_count": len(development_rows),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_query_count": len(duplicate_queries),
        "development_overlap_count": len(overlap_queries),
        "label_counts": dict(sorted(label_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "coverage": {
            "multi_intent_count": multi_intent_count,
            "context_count": context_count,
            "unsupported_count": unsupported_count,
            "entity_annotated_count": entity_annotated_count,
        },
        "minimum_label_count": minimum_label_count,
        "issue_count": len(issues),
        "issues": issues,
        "human_review_required": True,
        "note": (
            "Automatic audit checks structure and coverage only. It cannot "
            "certify semantic correctness of labels."
        ),
    }


def main() -> int:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    rows = load_jsonl(dataset_path)
    comparison_paths = [
        args.development_set,
        *args.comparison_set,
    ]
    development_rows = [
        row
        for relative_path in comparison_paths
        for row in load_jsonl(PROJECT_ROOT / relative_path)
    ]
    result = audit(
        rows,
        development_rows,
        dataset_path=dataset_path,
        minimum_label_count=args.minimum_label_count,
    )
    result["comparison_datasets"] = comparison_paths
    audit_path = write_json(PROJECT_ROOT / args.output, result)
    review_path = PROJECT_ROOT / args.review_output
    _write_review(review_path, rows, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved audit: {audit_path}")
    print(f"Saved review: {review_path}")
    return 0 if not result["issues"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
