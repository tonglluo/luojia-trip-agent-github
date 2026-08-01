"""Analyze frozen RAG retrieval and answer-generation reports."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import percentile, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval",
        default="evaluation/reports/rag_formal_v1_retrieval.json",
    )
    parser.add_argument(
        "--answers",
        default="evaluation/reports/rag_formal_v1_answers_attempt3.json",
    )
    parser.add_argument(
        "--invalid-attempt",
        action="append",
        default=[
            "evaluation/reports/rag_formal_v1_answers.json",
            "evaluation/reports/rag_formal_v1_answers_attempt2.json",
        ],
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_v1_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="evaluation/reports/rag_formal_v1_report.md",
    )
    return parser.parse_args()


def load(path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator if denominator else 0.0, 4)


def analyze(
    retrieval: dict[str, Any],
    answers: dict[str, Any],
    invalid_attempts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    if retrieval["dataset_sha256"] != answers["dataset_sha256"]:
        raise SystemExit("Retrieval and answer reports use different datasets")
    if retrieval["sample_count"] != answers["sample_count"]:
        raise SystemExit("Retrieval and answer sample counts differ")
    if retrieval["metrics"]["error_rate"] != 0:
        raise SystemExit("Retrieval report contains errors")
    if answers["metrics"]["error_rate"] != 0:
        raise SystemExit("Selected answer report is not error-free")

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers["rows"]:
        categories[str(row["category"])].append(row)

    category_rows = []
    for category, rows in sorted(categories.items()):
        answerable = [row for row in rows if row["answerable"]]
        unanswerable = [row for row in rows if not row["answerable"]]
        total_points = sum(
            int(row["key_points_total"] or 0) for row in answerable
        )
        covered_points = sum(
            int(row["key_points_covered"] or 0) for row in answerable
        )
        category_rows.append({
            "category": category,
            "sample_count": len(rows),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "key_point_coverage": (
                ratio(covered_points, total_points)
                if total_points
                else None
            ),
            "correct_refusal_rate": (
                ratio(
                    sum(bool(row["refused"]) for row in unanswerable),
                    len(unanswerable),
                )
                if unanswerable
                else None
            ),
        })

    answerable_rows = [
        row for row in answers["rows"] if row["answerable"]
    ]
    low_coverage_rows = sorted(
        (
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "covered": row["key_points_covered"],
                "total": row["key_points_total"],
                "coverage": ratio(
                    int(row["key_points_covered"]),
                    int(row["key_points_total"]),
                ),
            }
            for row in answerable_rows
            if row["key_points_covered"] < row["key_points_total"]
        ),
        key=lambda row: (row["coverage"], row["id"]),
    )
    incomplete_source_rows = [
        {
            "id": row["id"],
            "question": row["question"],
            "expected_sources": row["expected_sources"],
            "retrieved_sources": row["retrieved_sources"],
            "source_recall_at_5": row["source_recall_at_5"],
        }
        for row in retrieval["rows"]
        if row["answerable"] and not row["all_sources_recalled_at_5"]
    ]

    latencies = [
        float(row["latency_seconds"])
        for row in answers["rows"]
        if not row["error"]
    ]
    invalid_rows = []
    for path, report in invalid_attempts:
        errors = [
            row for row in report.get("rows", []) if row.get("error")
        ]
        error_counts: dict[str, int] = defaultdict(int)
        for row in errors:
            error_counts[str(row["error"])] += 1
        invalid_rows.append({
            "path": path,
            "dataset_sha256": report.get("dataset_sha256"),
            "sample_count": report.get("sample_count"),
            "error_rate": report.get("metrics", {}).get("error_rate"),
            "error_count": len(errors),
            "error_groups": dict(error_counts),
            "valid": False,
        })

    manual_answer_score = answers["metrics"].get(
        "manual_answer_score_mean",
    )
    manual_refusal_rate = answers["metrics"].get(
        "manual_correct_refusal_rate",
    )
    thresholds = {
        "recall_at_5": {
            "actual": retrieval["metrics"]["recall_at_5"],
            "threshold": 0.90,
            "passed": retrieval["metrics"]["recall_at_5"] >= 0.90,
        },
        "automatic_correct_refusal_rate": {
            "actual": round(
                1 - answers["metrics"][
                    "unanswerable_hallucination_rate"
                ],
                4,
            ),
            "threshold": 0.90,
            "passed": (
                1
                - answers["metrics"]["unanswerable_hallucination_rate"]
                >= 0.90
            ),
        },
        "manual_answer_score_mean": {
            "actual": manual_answer_score,
            "threshold": 1.60,
            "passed": (
                manual_answer_score >= 1.60
                if manual_answer_score is not None
                else None
            ),
        },
    }
    return {
        "generated_at": utc_now_iso(),
        "evaluation_type": "rag_frozen_v1_analysis",
        "dataset": answers["dataset"],
        "dataset_sha256": answers["dataset_sha256"],
        "sample_count": answers["sample_count"],
        "retrieval_report": retrieval["dataset"],
        "valid_answer_report": "evaluation/reports/"
        "rag_formal_v1_answers_attempt3.json",
        "metrics": {
            **retrieval["metrics"],
            "key_point_coverage": answers["metrics"][
                "key_point_coverage"
            ],
            "automatic_correct_refusal_rate": round(
                1 - answers["metrics"][
                    "unanswerable_hallucination_rate"
                ],
                4,
            ),
            "answer_error_rate": answers["metrics"]["error_rate"],
            "manual_answer_score_mean": manual_answer_score,
            "manual_correct_refusal_rate": manual_refusal_rate,
        },
        "threshold_checks": thresholds,
        "category_metrics": category_rows,
        "incomplete_source_rows": incomplete_source_rows,
        "low_substring_coverage_rows": low_coverage_rows,
        "valid_answer_latency": {
            "request_count": len(latencies),
            "mean_seconds": round(statistics.fmean(latencies), 4),
            "p50_seconds": round(float(percentile(latencies, 0.50)), 4),
            "p95_seconds": round(float(percentile(latencies, 0.95)), 4),
            "max_seconds": round(max(latencies), 4),
        },
        "invalid_attempts": invalid_rows,
        "manual_review_status": (
            "human_confirmed"
            if manual_answer_score is not None
            else "pending_human_confirmation"
        ),
        "caveats": [
            "Substring key-point coverage is diagnostic, not semantic accuracy.",
            "The automatic refusal metric uses phrase matching and still "
            "requires human review.",
            "Answer latency includes current provider/network variance and is "
            "not a controlled optimization benchmark.",
            "Two invalid attempts are retained and excluded from final quality "
            "metrics.",
        ],
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    manual_confirmed = result["manual_review_status"] == "human_confirmed"
    manual_score_cell = (
        f"{metrics['manual_answer_score_mean']:.2f}/2"
        if manual_confirmed
        else "待确认"
    )
    manual_score_status = "通过" if manual_confirmed else "待人工复核"
    lines = [
        "# RAG V1 正式评测报告",
        "",
        f"- 冻结集：`{result['dataset']}`",
        f"- SHA-256：`{result['dataset_sha256']}`",
        f"- 样本：{result['sample_count']}（50条可回答、10条不可回答）",
        "- 有效答案运行：Attempt 3，60/60 无错误",
        "",
        "## 核心结果",
        "",
        "|指标|结果|目标|状态|",
        "|---|---:|---:|---|",
        f"|Recall@5|{metrics['recall_at_5']:.2%}|≥90%|通过|",
        f"|MRR|{metrics['mrr']:.2%}|—|记录|",
        f"|任一来源命中率|{metrics['any_source_hit_at_5']:.2%}|—|记录|",
        f"|全来源召回率|{metrics['all_sources_recalled_at_5']:.2%}|—|记录|",
        f"|平均来源召回率|{metrics['mean_source_recall_at_5']:.2%}|—|记录|",
        f"|关键词覆盖率|{metrics['key_point_coverage']:.2%}|—|仅诊断|",
        f"|自动正确拒答率|"
        f"{metrics['automatic_correct_refusal_rate']:.2%}|≥90%|通过|",
        f"|人工答案均分|{manual_score_cell}|≥1.6/2|"
        f"{manual_score_status}|",
        (
            f"|人工正确拒答率|{metrics['manual_correct_refusal_rate']:.2%}|"
            "≥90%|通过|"
            if manual_confirmed
            else "|人工正确拒答率|待确认|≥90%|待人工复核|"
        ),
        "",
        "## 跨文档检索缺口",
        "",
        "Top-5 至少命中一个正确来源的比例为100%，但以下两题没有召回全部"
        "期望来源：",
        "",
        "|ID|问题|来源召回|",
        "|---|---|---:|",
    ]
    for row in result["incomplete_source_rows"]:
        lines.append(
            f"|{row['id']}|{row['question']}|"
            f"{row['source_recall_at_5']:.0%}|"
        )

    lines.extend([
        "",
        "## 无效尝试保留",
        "",
        "|尝试|错误数|错误率|原因|",
        "|---|---:|---:|---|",
    ])
    for index, attempt in enumerate(result["invalid_attempts"], 1):
        reasons = "；".join(
            f"{reason} × {count}"
            for reason, count in attempt["error_groups"].items()
        )
        lines.append(
            f"|{index}|{attempt['error_count']}|"
            f"{float(attempt['error_rate']):.2%}|{reasons}|"
        )

    latency = result["valid_answer_latency"]
    lines.extend([
        "",
        "## 运行与限制",
        "",
        f"- 有效答案延迟均值 / P50 / P95："
        f"{latency['mean_seconds']:.2f}s / {latency['p50_seconds']:.2f}s / "
        f"{latency['p95_seconds']:.2f}s。",
        "- 关键词覆盖率不等同于答案准确率；同义表达和格式差异会造成漏计。",
        (
            "- 不可回答题已经人工确认；10条均为正确拒答。"
            if manual_confirmed
            else (
                "- 自动拒答率仍需人工确认，尤其是知识库出现相似词但缺少"
                "具体制度的硬负样本。"
            )
        ),
        (
            "- 人工答案评分已经项目负责人逐条确认，可作为 RAG V1 "
            "最终质量结果。"
            if manual_confirmed
            else (
                "- 当前仍不能对外报告人工答案均分，直到项目负责人确认"
                "评分表。"
            )
        ),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    retrieval = load(args.retrieval)
    answers = load(args.answers)
    invalid_attempts = [
        (path, load(path))
        for path in args.invalid_attempt
        if (PROJECT_ROOT / path).exists()
    ]
    result = analyze(retrieval, answers, invalid_attempts)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps(result["threshold_checks"], ensure_ascii=False, indent=2))
    print(f"Saved analysis: {output_path}")
    print(f"Saved report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
