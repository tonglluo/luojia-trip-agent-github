"""Build an evidence-grounded AI pre-review for the frozen RAG V1 run.

This script does not edit model answers and does not produce human scores.
Every answerable row is tied to the exact Top-5 chunks reconstructed for the
error-free Attempt 3 run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso, write_json


EVIDENCE_RANKS = {
    "rag_formal_001": [1, 2],
    "rag_formal_002": [1],
    "rag_formal_003": [1, 2],
    "rag_formal_004": [1],
    "rag_formal_005": [1],
    "rag_formal_006": [1],
    "rag_formal_007": [1],
    "rag_formal_008": [1],
    "rag_formal_009": [1],
    "rag_formal_010": [1],
    "rag_formal_011": [1, 2, 3],
    "rag_formal_012": [1, 3],
    "rag_formal_013": [1],
    "rag_formal_014": [1, 2, 4],
    "rag_formal_015": [1, 2],
    "rag_formal_016": [1],
    "rag_formal_017": [1],
    "rag_formal_018": [1],
    "rag_formal_019": [1],
    "rag_formal_020": [1],
    "rag_formal_021": [1],
    "rag_formal_022": [1],
    "rag_formal_023": [1, 2],
    "rag_formal_024": [1],
    "rag_formal_025": [1],
    "rag_formal_026": [1, 4],
    "rag_formal_027": [1, 2, 3],
    "rag_formal_028": [1],
    "rag_formal_029": [2],
    "rag_formal_030": [1],
    "rag_formal_031": [1],
    "rag_formal_032": [1],
    "rag_formal_033": [1],
    "rag_formal_034": [1, 2, 4],
    "rag_formal_035": [2],
    "rag_formal_036": [1, 2],
    "rag_formal_037": [1, 2],
    "rag_formal_038": [1],
    "rag_formal_039": [1],
    "rag_formal_040": [1],
    "rag_formal_041": [1, 2],
    "rag_formal_042": [1],
    "rag_formal_043": [2, 3, 4],
    "rag_formal_044": [2],
    "rag_formal_045": [3],
    "rag_formal_046": [1],
    "rag_formal_047": [1, 2, 4],
    "rag_formal_048": [1, 2],
    "rag_formal_049": [1, 2],
    "rag_formal_050": [1, 2, 3],
}


ISSUES = {
    "rag_formal_030": {
        "score": 1,
        "kind": "incomplete_answer",
        "detail": (
            "题目询问“为什么不能乘电梯”，回答只重述应走消防通道，"
            "没有解释原因；Top-5 本身也只给出动作要求、没有原因证据，"
            "因此不能按完整回答计 2 分。"
        ),
    },
    "rag_formal_034": {
        "score": 1,
        "kind": "conflicting_evidence",
        "detail": (
            "回答同时给出 10%-30% 与 10%-20% 两个优惠幅度。两者分别"
            "受 Top-2 和 Top-4 支持，但知识库口径冲突且回答未给出统一"
            "适用范围。"
        ),
    },
    "rag_formal_035": {
        "score": 1,
        "kind": "missing_key_points",
        "detail": (
            "现有回答受 Top-2 支持，但遗漏冻结答案要点中的自动推荐合规"
            "产品、筛选超标产品、超标提醒/预警、自动关联审批流程。"
        ),
    },
    "rag_formal_037": {
        "score": 1,
        "kind": "missing_key_point",
        "detail": "回答遗漏流程中的“提交审批”步骤。",
    },
    "rag_formal_046": {
        "score": 1,
        "kind": "missing_key_point",
        "detail": (
            "回答覆盖事前申请、说明理由和按平台价格报销，但遗漏冻结"
            "答案要点中的正式发票要求。"
        ),
    },
    "rag_formal_047": {
        "score": 1,
        "kind": "missing_key_points",
        "detail": (
            "回答覆盖延误证明、改签凭证、住宿/餐饮发票和说明，但遗漏"
            "住宿发票应体现的住宿日期、天数和金额。"
        ),
    },
    "rag_formal_050": {
        "score": 1,
        "kind": "unsupported_or_misleading_claim",
        "detail": (
            "“鼓励使用高铁而非飞机（或汽车）”中的“或汽车”不成立。"
            "Top-3 表述为 300 公里以内建议选择高铁或汽车，并未建议"
            "高铁优先于汽车。"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        default="evaluation/reports/rag_formal_v1_answers_attempt3.json",
    )
    parser.add_argument(
        "--context",
        default="evaluation/reports/rag_formal_v1_retrieval_context.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_v1_grounded_pre_review.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="evaluation/reports/rag_formal_v1_grounded_pre_review.md",
    )
    return parser.parse_args()


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def evidence_for(
    context_row: dict[str, Any],
    ranks: list[int],
) -> list[dict[str, Any]]:
    documents = {document["rank"]: document for document in context_row["documents"]}
    evidence = []
    for rank in ranks:
        if rank not in documents:
            raise ValueError(f"{context_row['id']}: missing evidence rank {rank}")
        document = documents[rank]
        evidence.append({
            "rank": rank,
            "source": document["source"],
            "chunk_index": document["chunk_index"],
            "content": document["content"],
        })
    return evidence


def build(
    answer_report: dict[str, Any],
    context_report: dict[str, Any],
) -> dict[str, Any]:
    if answer_report["metrics"]["error_rate"] != 0:
        raise ValueError("Grounded review requires an error-free answer run")
    if not all(row["sources_match_saved_run"] for row in context_report["rows"]):
        raise ValueError("Reconstructed Top-5 does not match the saved answer run")

    context_rows = {row["id"]: row for row in context_report["rows"]}
    rows = []
    scores = []
    correct_refusals = 0
    unanswerable_count = 0

    answerable_ids = {
        row["id"] for row in answer_report["rows"] if row["answerable"]
    }
    if answerable_ids != set(EVIDENCE_RANKS):
        missing = sorted(answerable_ids - set(EVIDENCE_RANKS))
        extra = sorted(set(EVIDENCE_RANKS) - answerable_ids)
        raise ValueError(f"Evidence map mismatch: missing={missing}, extra={extra}")

    for row in answer_report["rows"]:
        context_row = context_rows[row["id"]]
        if row["answerable"]:
            issue = ISSUES.get(row["id"])
            score = issue["score"] if issue else 2
            scores.append(score)
            ranks = EVIDENCE_RANKS[row["id"]]
            evidence = evidence_for(context_row, ranks)
            grounding_status = (
                issue["kind"] if issue else "all_concrete_claims_supported"
            )
            rationale = (
                issue["detail"]
                if issue
                else (
                    "逐项核对后，答案中的具体事实可在列出的当次 Top-5 "
                    "片段中找到，且未发现遗漏、冲突或相反含义。"
                )
            )
            outcome = None
        else:
            score = None
            evidence = []
            grounding_status = "correct_refusal" if row["refused"] else "hallucination"
            outcome = grounding_status
            unanswerable_count += 1
            correct_refusals += outcome == "correct_refusal"
            rationale = (
                "知识库无可支持该问题具体答案的证据，模型明确拒答且未"
                "给出制度、金额或流程性结论。"
                if outcome == "correct_refusal"
                else "知识库无答案时仍输出了具体结论。"
            )

        rows.append({
            "id": row["id"],
            "question": row["question"],
            "answerable": row["answerable"],
            "answer_key_points": row["answer_key_points"],
            "answer": row["answer"],
            "retrieved_sources": row["retrieved_sources"],
            "top5_sources_match_saved_run": context_row["sources_match_saved_run"],
            "grounding_status": grounding_status,
            "evidence": evidence,
            "suggested_answer_score_0_to_2": score,
            "suggested_unanswerable_outcome": outcome,
            "review_rationale": rationale,
            "human_decision": None,
            "human_note": "",
        })

    return {
        "review_type": "evidence_grounded_ai_pre_review_not_human",
        "generated_at": utc_now_iso(),
        "source_answer_report": str(answer_report.get("dataset", "")),
        "source_context_report": "rag_formal_v1_retrieval_context.json",
        "dataset_sha256": answer_report["dataset_sha256"],
        "sample_count": len(rows),
        "method": {
            "evidence_boundary": (
                "Only the exact Top-5 chunks reconstructed for Attempt 3 are "
                "accepted as grounding evidence."
            ),
            "answer_mutation": "Original generated answers were not edited.",
            "score_2_rule": (
                "Required content is complete and every concrete factual claim "
                "is supported without contradiction."
            ),
        },
        "supersedes": [
            "evaluation/reports/rag_formal_v1_ai_pre_review.json",
            "evaluation/reports/rag_formal_v1_ai_pre_review.md",
        ],
        "suggested_metrics": {
            "answer_score_mean": round(sum(scores) / len(scores), 4),
            "correct_refusal_rate": round(
                correct_refusals / unanswerable_count,
                4,
            ),
            "score_2_count": scores.count(2),
            "score_1_count": scores.count(1),
            "score_0_count": scores.count(0),
        },
        "human_review_required": True,
        "rows": rows,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["suggested_metrics"]
    lines = [
        "# RAG V1 证据落地预审（修订版）",
        "",
        "> 这仍是 AI 辅助预审，不是人工最终评分。原始模型答案未被修改。",
        "",
        "## 口径",
        "",
        "- 唯一证据边界：Attempt 3 当次实际检索的 Top-5 片段。",
        "- 2 分：内容完整，且每项具体事实均有证据、无冲突。",
        "- 1 分：部分正确，但有遗漏、歧义、口径冲突或误导性扩展。",
        "- 0 分：核心结论错误、编造或答非所问。",
        "",
        "## 修订建议",
        "",
        f"- 样本：{result['sample_count']}（可回答 50，不可回答 10）",
        f"- 建议答案均分：{metrics['answer_score_mean']:.2f}/2",
        f"- 建议 2 / 1 / 0 分：{metrics['score_2_count']} / "
        f"{metrics['score_1_count']} / {metrics['score_0_count']}",
        f"- 不可回答题正确拒答率：{metrics['correct_refusal_rate']:.2%}",
        "",
        "## 需要重点复核的 1 分项",
        "",
        "|ID|问题类型|判定依据|",
        "|---|---|---|",
    ]
    for row in result["rows"]:
        if row["suggested_answer_score_0_to_2"] == 1:
            lines.append(
                f"|{row['id']}|{row['grounding_status']}|"
                f"{escape(row['review_rationale'])}|"
            )

    lines.extend([
        "",
        "## 全量逐条证据索引",
        "",
        "|确认|ID|建议|证据片段|核验结论|人工调整|",
        "|---|---|---:|---|---|---|",
    ])
    for row in result["rows"]:
        suggestion = (
            str(row["suggested_answer_score_0_to_2"])
            if row["answerable"]
            else row["suggested_unanswerable_outcome"]
        )
        evidence = (
            "；".join(
                f"Top-{item['rank']} {item['source']}#chunk-{item['chunk_index']}"
                for item in row["evidence"]
            )
            or "无答案题：核验拒答"
        )
        lines.append(
            f"|[ ]|{row['id']}|{suggestion}|{escape(evidence)}|"
            f"{escape(row['review_rationale'])}|________|"
        )

    lines.extend([
        "",
        "## 用户点名扩展的核验",
        "",
        "- `rag_formal_012`“长期保存”：Top-3，"
        "`02_reimbursement_policy.txt#chunk-27`。",
        "- `rag_formal_013`“到账后短信或邮件通知”：Top-1，"
        "`02_reimbursement_policy.txt#chunk-13`。",
        "- `rag_formal_015`“超过500元需更详细证明”：Top-1，"
        "`04_faq.txt#chunk-14`。",
        "- `rag_formal_029` 客服电话 `400-800-8888`：Top-2，"
        "`05_emergency_procedures.txt#chunk-11`。",
        "",
        "以上四项均受当次 Top-5 支持，保留答案并不构成事实扩展错误。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    answers_path = PROJECT_ROOT / args.answers
    context_path = PROJECT_ROOT / args.context
    answer_report = json.loads(answers_path.read_text(encoding="utf-8"))
    context_report = json.loads(context_path.read_text(encoding="utf-8"))
    result = build(answer_report, context_report)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps(result["suggested_metrics"], ensure_ascii=False, indent=2))
    print(f"Saved grounded pre-review: {output_path}")
    print(f"Saved grounded review sheet: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
