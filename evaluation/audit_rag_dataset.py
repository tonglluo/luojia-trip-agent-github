"""Audit a candidate RAG evaluation dataset before it is frozen."""
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


KNOWLEDGE_DIR = (
    PROJECT_ROOT
    / ".claude"
    / "skills"
    / "ask-question"
    / "data"
    / "documents"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/rag_eval.formal.candidate.jsonl",
    )
    parser.add_argument(
        "--comparison-set",
        action="append",
        default=["evaluation/datasets/rag_eval.sample.jsonl"],
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_candidate_audit.json",
    )
    parser.add_argument(
        "--review-output",
        default="evaluation/reports/rag_formal_review.md",
    )
    parser.add_argument("--minimum-answerable", type=int, default=50)
    parser.add_argument("--minimum-unanswerable", type=int, default=10)
    return parser.parse_args()


def normalize(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value).casefold())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    *,
    dataset_path: Path,
    minimum_answerable: int,
    minimum_unanswerable: int,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    available_sources = {
        path.name: path
        for path in KNOWLEDGE_DIR.glob("*.txt")
    }
    source_text = {
        name: normalize(path.read_text(encoding="utf-8"))
        for name, path in available_sources.items()
    }

    comparison_questions = {
        normalize(row.get("question", ""))
        for row in comparison_rows
    }
    questions = [normalize(row.get("question", "")) for row in rows]
    duplicate_questions = sorted(
        question
        for question, count in Counter(questions).items()
        if question and count > 1
    )
    overlap_questions = sorted(set(questions) & comparison_questions)

    for question in duplicate_questions:
        issues.append({
            "scope": question,
            "code": "duplicate_question",
            "message": "归一化后的问题在候选集中重复。",
        })
    for question in overlap_questions:
        issues.append({
            "scope": question,
            "code": "comparison_overlap",
            "message": "候选问题与已有评测问题完全重复。",
        })

    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    answerable_count = 0
    unanswerable_count = 0
    multi_source_count = 0

    required_fields = {
        "id",
        "category",
        "question",
        "answerable",
        "expected_sources",
        "answer_key_points",
    }
    for index, row in enumerate(rows, 1):
        row_id = str(row.get("id", f"row_{index}"))
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            issues.append({
                "scope": row_id,
                "code": "missing_fields",
                "message": f"缺少字段：{missing_fields}",
            })
            continue

        category_counts.update([str(row["category"])])
        answerable = row["answerable"]
        sources = row["expected_sources"]
        key_points = row["answer_key_points"]

        if not isinstance(answerable, bool):
            issues.append({
                "scope": row_id,
                "code": "invalid_answerable",
                "message": "answerable 必须是布尔值。",
            })
            continue
        if not isinstance(sources, list) or not all(
            isinstance(item, str) and item.strip() for item in sources
        ):
            issues.append({
                "scope": row_id,
                "code": "invalid_expected_sources",
                "message": "expected_sources 必须是非空字符串数组或空数组。",
            })
            continue
        if not isinstance(key_points, list) or not all(
            isinstance(item, str) and item.strip() for item in key_points
        ):
            issues.append({
                "scope": row_id,
                "code": "invalid_key_points",
                "message": "answer_key_points 必须是非空字符串数组或空数组。",
            })
            continue

        unknown_sources = sorted(set(sources) - set(available_sources))
        if unknown_sources:
            issues.append({
                "scope": row_id,
                "code": "unknown_source",
                "message": f"知识库中不存在来源：{unknown_sources}",
            })

        if answerable:
            answerable_count += 1
            source_counts.update(set(sources))
            multi_source_count += len(set(sources)) > 1
            if not sources:
                issues.append({
                    "scope": row_id,
                    "code": "answerable_without_source",
                    "message": "可回答样本必须至少标注一个来源。",
                })
            if not key_points:
                issues.append({
                    "scope": row_id,
                    "code": "answerable_without_key_points",
                    "message": "可回答样本必须至少标注一个答案要点。",
                })

            combined_source_text = "".join(
                source_text.get(source, "") for source in sources
            )
            unsupported_points = [
                point
                for point in key_points
                if normalize(point) not in combined_source_text
            ]
            if unsupported_points:
                warnings.append({
                    "scope": row_id,
                    "code": "key_point_not_exact_substring",
                    "message": (
                        "以下要点不是来源原文的精确子串，需人工确认其为可靠同义归纳："
                        f"{unsupported_points}"
                    ),
                })
        else:
            unanswerable_count += 1
            if sources or key_points:
                issues.append({
                    "scope": row_id,
                    "code": "unanswerable_has_answer",
                    "message": "不可回答样本的来源和答案要点必须为空。",
                })
            if not str(row.get("annotation_note", "")).strip():
                issues.append({
                    "scope": row_id,
                    "code": "unanswerable_without_rationale",
                    "message": "不可回答样本必须说明知识边界与拒答依据。",
                })

    if answerable_count < minimum_answerable:
        issues.append({
            "scope": "dataset",
            "code": "insufficient_answerable",
            "message": (
                f"可回答样本 {answerable_count} 少于要求 {minimum_answerable}。"
            ),
        })
    if unanswerable_count < minimum_unanswerable:
        issues.append({
            "scope": "dataset",
            "code": "insufficient_unanswerable",
            "message": (
                f"不可回答样本 {unanswerable_count} 少于要求 "
                f"{minimum_unanswerable}。"
            ),
        })
    uncovered_sources = sorted(set(available_sources) - set(source_counts))
    if uncovered_sources:
        issues.append({
            "scope": "dataset",
            "code": "source_not_covered",
            "message": f"以下知识文档没有正例覆盖：{uncovered_sources}",
        })

    return {
        "generated_at": utc_now_iso(),
        "status": "pass" if not issues else "needs_review",
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256(dataset_path),
        "sample_count": len(rows),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "multi_source_count": multi_source_count,
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "available_sources": sorted(available_sources),
        "duplicate_question_count": len(duplicate_questions),
        "comparison_overlap_count": len(overlap_questions),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "human_review_required": True,
        "note": (
            "自动审计只验证结构、覆盖和可追溯性；答案语义与不可回答边界"
            "仍需在首次模型评测前人工复核。"
        ),
    }


def write_review(
    path: Path,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    warning_ids = {
        warning["scope"] for warning in result["warnings"]
    }
    lines = [
        "# RAG 正式评测集人工复核清单",
        "",
        "- 状态：候选集，尚未冻结",
        f"- 样本数：{len(rows)}",
        f"- 可回答 / 不可回答：{result['answerable_count']} / "
        f"{result['unanswerable_count']}",
        f"- 多来源问题：{result['multi_source_count']}",
        f"- SHA-256：`{result['dataset_sha256']}`",
        f"- 自动审计问题 / 警告：{result['issue_count']} / "
        f"{result['warning_count']}",
        "",
        "逐条确认问题可读、答案要点被指定来源支持、不可回答问题确实不在"
        "八份知识文档中。带“需核同义”的项目须重点核对。",
        "",
        "|确认|ID|类别|问题|可回答|期望来源|答案要点|备注|",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        question = str(row["question"]).replace("|", "\\|")
        sources = "<br>".join(row["expected_sources"]) or "—"
        key_points = "；".join(row["answer_key_points"]).replace("|", "\\|") or "—"
        note_parts = []
        if row["id"] in warning_ids:
            note_parts.append("需核同义归纳")
        if row.get("annotation_note"):
            note_parts.append(str(row["annotation_note"]))
        note = "；".join(note_parts).replace("|", "\\|")
        lines.append(
            f"|[ ]|{row['id']}|{row['category']}|{question}|"
            f"{'是' if row['answerable'] else '否'}|{sources}|"
            f"{key_points}|{note}|"
        )
    lines.extend([
        "",
        "## 冻结前签字",
        "",
        "- [ ] 60 条问题已逐条复核",
        "- [ ] 50 条可回答样本的来源与答案要点准确",
        "- [ ] 10 条不可回答样本未被知识库覆盖",
        "- [ ] 5 条跨文档问题确实需要多个来源",
        "- [ ] 未根据模型预测结果修改任何标注",
        "- 复核人：________________",
        "- 日期：__________________",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    rows = load_jsonl(dataset_path)
    comparison_rows = [
        row
        for relative_path in args.comparison_set
        for row in load_jsonl(PROJECT_ROOT / relative_path)
    ]
    result = audit(
        rows,
        comparison_rows,
        dataset_path=dataset_path,
        minimum_answerable=args.minimum_answerable,
        minimum_unanswerable=args.minimum_unanswerable,
    )
    result["comparison_datasets"] = args.comparison_set
    audit_path = write_json(PROJECT_ROOT / args.output, result)
    review_path = PROJECT_ROOT / args.review_output
    write_review(review_path, rows, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved audit: {audit_path}")
    print(f"Saved review: {review_path}")
    return 0 if not result["issues"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
