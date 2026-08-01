"""Prepare an AI-assisted RAG answer review without claiming human scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso, write_json


PARTIAL_CASES = {
    "rag_formal_034": (
        "回答同时给出平台文档的10%-30%和FAQ的10%-20%，如实暴露了"
        "知识库口径冲突，但没有为用户明确统一口径。"
    ),
    "rag_formal_035": (
        "回答只给出泛化的标注、审核和咨询方式，遗漏自动推荐合规产品、"
        "筛选超标产品、超标预警和自动关联审批四项核心机制。"
    ),
    "rag_formal_037": (
        "回答覆盖关联申请、自动导入、补充费用和上传发票，但遗漏最终"
        "提交审批步骤。"
    ),
    "rag_formal_046": (
        "回答覆盖提前申请、说明理由和按平台价格报销，但遗漏正式发票要求。"
    ),
    "rag_formal_047": (
        "回答覆盖延误证明、住宿发票及说明凭证，但因报销制度来源未进入"
        "Top-5，遗漏发票上的住宿日期、天数和金额。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="evaluation/reports/rag_formal_v1_answers_attempt3.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_v1_ai_pre_review.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="evaluation/reports/rag_formal_v1_ai_pre_review.md",
    )
    return parser.parse_args()


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def build(report: dict[str, Any]) -> dict[str, Any]:
    if report["metrics"]["error_rate"] != 0:
        raise SystemExit("AI pre-review requires an error-free answer report")
    rows = []
    answer_scores = []
    correct_refusals = 0
    unanswerable_count = 0

    for row in report["rows"]:
        if row["answerable"]:
            score = 1 if row["id"] in PARTIAL_CASES else 2
            rationale = PARTIAL_CASES.get(
                row["id"],
                (
                    "回答覆盖问题所需语义，未发现与指定知识来源冲突的事实。"
                    "关键词未全部命中时，主要是格式或同义表达差异。"
                ),
            )
            answer_scores.append(score)
            outcome = None
        else:
            score = None
            outcome = (
                "correct_refusal"
                if row["refused"]
                else "hallucination"
            )
            correct_refusals += outcome == "correct_refusal"
            unanswerable_count += 1
            rationale = (
                "回答明确表示知识库没有相关信息，未提供具体制度或数值。"
                if outcome == "correct_refusal"
                else "回答在知识库没有答案时仍给出了具体结论。"
            )
        rows.append({
            "id": row["id"],
            "question": row["question"],
            "answerable": row["answerable"],
            "expected_sources": row["expected_sources"],
            "answer_key_points": row["answer_key_points"],
            "answer": row["answer"],
            "substring_key_points_covered": row["key_points_covered"],
            "substring_key_points_total": row["key_points_total"],
            "suggested_answer_score_0_to_2": score,
            "suggested_unanswerable_outcome": outcome,
            "ai_review_rationale": rationale,
            "human_decision": None,
            "human_note": "",
        })

    return {
        "review_type": "ai_pre_review_not_human",
        "generated_at": utc_now_iso(),
        "source_report": report["dataset"],
        "dataset_sha256": report["dataset_sha256"],
        "sample_count": len(rows),
        "suggested_metrics": {
            "answer_score_mean": round(
                sum(answer_scores) / len(answer_scores),
                4,
            ),
            "correct_refusal_rate": round(
                correct_refusals / unanswerable_count,
                4,
            ),
            "score_2_count": answer_scores.count(2),
            "score_1_count": answer_scores.count(1),
            "score_0_count": answer_scores.count(0),
        },
        "human_review_required": True,
        "warning": (
            "These are AI-assisted suggestions, not human scores. A human "
            "reviewer must approve or revise every decision before finalization."
        ),
        "rows": rows,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["suggested_metrics"]
    lines = [
        "# RAG V1 答案 AI 预审",
        "",
        "> 本文件不是人工评分结果。建议分数必须由项目负责人逐条确认或修改。",
        "",
        f"- 样本：{result['sample_count']}",
        f"- 建议答案均分：{metrics['answer_score_mean']:.2f}/2",
        f"- 建议正确拒答率：{metrics['correct_refusal_rate']:.2%}",
        f"- 建议 2 / 1 / 0 分数量：{metrics['score_2_count']} / "
        f"{metrics['score_1_count']} / {metrics['score_0_count']}",
        "",
        "|确认|ID|问题|答案|要点覆盖|建议|预审理由|人工调整|",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in result["rows"]:
        coverage = (
            "拒答题"
            if not row["answerable"]
            else (
                f"{row['substring_key_points_covered']}/"
                f"{row['substring_key_points_total']}"
            )
        )
        suggestion = (
            str(row["suggested_answer_score_0_to_2"])
            if row["answerable"]
            else row["suggested_unanswerable_outcome"]
        )
        lines.append(
            f"|[ ]|{row['id']}|{escape(row['question'])}|"
            f"{escape(row['answer'])}|{coverage}|{suggestion}|"
            f"{escape(row['ai_review_rationale'])}|________|"
        )
    lines.extend([
        "",
        "## 人工确认规则",
        "",
        "- 2 分：关键内容完整，无事实错误。",
        "- 1 分：部分正确，但存在明显遗漏、歧义或口径冲突。",
        "- 0 分：错误、编造或答非所问。",
        "- 不可回答题：标记 `correct_refusal`、`hallucination` 或 `ambiguous`。",
        "- 不能仅凭关键词覆盖率评分；必须阅读问题、答案和来源证据。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = json.loads(
        (PROJECT_ROOT / args.report).read_text(encoding="utf-8"),
    )
    result = build(report)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps(result["suggested_metrics"], ensure_ascii=False, indent=2))
    print(f"Saved AI pre-review: {output_path}")
    print(f"Saved review sheet: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
