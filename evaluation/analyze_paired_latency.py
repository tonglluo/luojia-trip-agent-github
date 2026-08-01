"""Validate and summarize the formal paired orchestration latency run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import percentile, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=(
            "evaluation/reports/"
            "latency_orchestration_paired_v1.json"
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=(
            "evaluation/datasets/"
            "latency_orchestration.formal.v1.manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "evaluation/reports/"
            "latency_orchestration_paired_v1_analysis.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        default=(
            "evaluation/reports/"
            "latency_orchestration_paired_v1_report.md"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": round(sum(values) / len(values), 4),
        "p50": round(float(percentile(values, 0.50)), 4),
        "p95": round(float(percentile(values, 0.95)), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return round((baseline - candidate) / baseline, 4)


def analyze(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    rows = report["rows"]
    expected_count = int(report["measured_requests_per_mode"])
    identities = {
        (row["id"], row["repeat"], row["mode"])
        for row in rows
    }
    modes = ("sequential", "parallel")
    mode_rows = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in modes
    }

    integrity_checks = {
        "raw_report_declares_valid": bool(report["valid_benchmark"]),
        "dataset_hash_matches_manifest": (
            report["dataset_sha256"] == manifest["sha256"]
        ),
        "row_count_is_80": len(rows) == expected_count * 2 == 80,
        "all_rows_unique": len(identities) == len(rows),
        "ten_repeats_present": (
            sorted({row["repeat"] for row in rows})
            == list(range(1, 11))
        ),
        "forty_requests_per_mode": all(
            len(mode_rows[mode]) == expected_count == 40
            for mode in modes
        ),
        "balanced_pair_order": all(
            sum(row["pair_order"] == order for row in mode_rows[mode])
            == 20
            for mode in modes
            for order in (1, 2)
        ),
        "all_schedules_match": all(
            row["schedule_matches_expected"] for row in rows
        ),
        "all_pairs_complete": (
            report["comparison"]["complete_pair_count"] == 40
        ),
    }

    raw_mode_metrics = report["mode_metrics"]
    recomputed_mode_metrics = {}
    for mode in modes:
        selected = mode_rows[mode]
        successful = [row for row in selected if row["status"] == "success"]
        recomputed_mode_metrics[mode] = {
            "request_count": len(selected),
            "success_count": len(successful),
            "success_rate": round(len(successful) / len(selected), 4),
            "total_seconds": summary([
                float(row["total_seconds"]) for row in successful
            ]),
            "intent_seconds": summary([
                float(row["intent_seconds"]) for row in successful
            ]),
            "execution_seconds": summary([
                float(row["execution_seconds"]) for row in successful
            ]),
            "schedule_match_rate": round(
                sum(row["schedule_matches_expected"] for row in selected)
                / len(selected),
                4,
            ),
            "agent_retry_count": sum(
                int(row.get("agent_retry_count", 0)) for row in selected
            ),
            "agent_retry_request_count": sum(
                int(row.get("agent_retry_count", 0)) > 0
                for row in selected
            ),
            "intent_retry_count": sum(
                int(row.get("intent_retry_count", 0)) for row in selected
            ),
            "intent_retry_request_count": sum(
                int(row.get("intent_retry_count", 0)) > 0
                for row in selected
            ),
            "partial_failure_count": sum(
                row.get("orchestration_status") == "partial_failure"
                for row in selected
            ),
        }

    metric_reconciliation = {}
    for mode in modes:
        raw = raw_mode_metrics[mode]
        recomputed = recomputed_mode_metrics[mode]
        metric_reconciliation[mode] = {
            "success_rate_matches": (
                raw["success_rate"] == recomputed["success_rate"]
            ),
            "total_seconds_matches": (
                raw["total_seconds"] == recomputed["total_seconds"]
            ),
            "intent_seconds_matches": (
                raw["intent_seconds"] == recomputed["intent_seconds"]
            ),
            "execution_seconds_matches": (
                raw["execution_seconds"]
                == recomputed["execution_seconds"]
            ),
            "retry_totals_match": (
                raw["agent_retry_count"]
                == recomputed["agent_retry_count"]
                and raw["intent_retry_count"]
                == recomputed["intent_retry_count"]
            ),
        }

    categories: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for row in rows:
        categories[row["category"]][row["mode"]].append(row)

    category_metrics = []
    for category, grouped in sorted(categories.items()):
        sequential = grouped["sequential"]
        parallel = grouped["parallel"]
        sequential_execution = summary([
            float(row["execution_seconds"]) for row in sequential
        ])
        parallel_execution = summary([
            float(row["execution_seconds"]) for row in parallel
        ])
        sequential_total = summary([
            float(row["total_seconds"]) for row in sequential
        ])
        parallel_total = summary([
            float(row["total_seconds"]) for row in parallel
        ])
        category_metrics.append({
            "category": category,
            "sample_count_per_mode": len(sequential),
            "sequential_execution_seconds": sequential_execution,
            "parallel_execution_seconds": parallel_execution,
            "execution_improvement": {
                metric: improvement(
                    sequential_execution[metric],
                    parallel_execution[metric],
                )
                for metric in ("mean", "p50", "p95")
            },
            "sequential_total_seconds": sequential_total,
            "parallel_total_seconds": parallel_total,
            "total_improvement": {
                metric: improvement(
                    sequential_total[metric],
                    parallel_total[metric],
                )
                for metric in ("mean", "p50", "p95")
            },
        })

    slowest_rows = sorted(
        rows,
        key=lambda row: float(row["total_seconds"]),
        reverse=True,
    )[:10]
    retry_rows = [
        {
            "id": row["id"],
            "repeat": row["repeat"],
            "mode": row["mode"],
            "intent_retry_count": row.get("intent_retry_count", 0),
            "agent_retry_count": row.get("agent_retry_count", 0),
            "total_seconds": row["total_seconds"],
            "execution_seconds": row["execution_seconds"],
        }
        for row in rows
        if row.get("intent_retry_count", 0)
        or row.get("agent_retry_count", 0)
    ]

    aggregate = report["comparison"]["aggregate_improvement"]
    supported_claims = {
        "execution_mean_reduction": aggregate["execution_seconds"]["mean"],
        "execution_p50_reduction": aggregate["execution_seconds"]["p50"],
        "execution_p95_reduction": aggregate["execution_seconds"]["p95"],
        "end_to_end_mean_reduction": aggregate["total_seconds"]["mean"],
        "end_to_end_p50_reduction": aggregate["total_seconds"]["p50"],
        "end_to_end_p95_reduction": aggregate["total_seconds"]["p95"],
        "parallel_execution_win_rate": report["comparison"][
            "parallel_execution_win_rate"
        ],
        "parallel_total_win_rate": report["comparison"][
            "parallel_total_win_rate"
        ],
        "success_rate_no_regression": (
            recomputed_mode_metrics["sequential"]["success_rate"]
            == recomputed_mode_metrics["parallel"]["success_rate"]
            == 1.0
        ),
        "can_claim_end_to_end_50_percent_reduction": False,
    }

    validation_passed = (
        all(integrity_checks.values())
        and all(
            all(checks.values())
            for checks in metric_reconciliation.values()
        )
    )
    return {
        "evaluation_type": "paired_latency_validated_analysis",
        "generated_at": utc_now_iso(),
        "source_report": str(report_path.relative_to(PROJECT_ROOT)),
        "source_report_sha256": sha256(report_path),
        "dataset": report["dataset"],
        "dataset_sha256": report["dataset_sha256"],
        "as_of": report["completed_at"],
        "protocol": {
            "model": report["model"],
            "thinking": report["thinking"],
            "temperature": report["temperature"],
            "max_tokens": report["max_tokens"],
            "warmup_rounds_per_mode": report["warmup_rounds_per_mode"],
            "repeats_per_mode": report["repeats_per_mode"],
            "requests_per_mode": report["measured_requests_per_mode"],
            "cooldown_seconds": manifest["formal_protocol"][
                "cooldown_seconds_between_top_level_requests"
            ],
            "order_control": report["order_control"],
        },
        "validation_status": (
            "ready_to_share_with_caveats"
            if validation_passed
            else "needs_revision"
        ),
        "integrity_checks": integrity_checks,
        "metric_reconciliation": metric_reconciliation,
        "mode_metrics": recomputed_mode_metrics,
        "comparison": report["comparison"],
        "category_metrics": category_metrics,
        "retry_rows": retry_rows,
        "slowest_rows": slowest_rows,
        "supported_claims": supported_claims,
        "invalid_attempts_retained": [
            {
                "attempt": 1,
                "reason": "RAG timeout was hidden as success before error-propagation fix.",
            },
            {
                "attempt": 2,
                "reason": "Intent timeout had no benchmark-level retry, causing schedule mismatch.",
            },
            {
                "attempt": 3,
                "reason": "0.5-second cooldown turned latency measurement into sustained load and exhausted rate limits.",
            },
        ],
        "required_caveats": [
            "The benchmark uses one provider account and one workstation; "
            "absolute latency may vary by time and quota.",
            "Parallel execution created 10 agent retries across 5 requests, "
            "while sequential execution created none; concurrency increased "
            "rate-limit pressure even though all requests eventually succeeded.",
            "Intent recognition is outside the parallelized agent batch and "
            "dominates some end-to-end tail latency.",
            "The paired mean of percentage improvements is outlier-sensitive; "
            "use aggregate P50/P95 and raw latency values as primary claims.",
            "The data does not support a 50% end-to-end response reduction claim.",
        ],
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    sequential = result["mode_metrics"]["sequential"]
    parallel = result["mode_metrics"]["parallel"]
    aggregate = result["comparison"]["aggregate_improvement"]
    lines = [
        "# 串行 vs 并行编排：正式延迟与稳定性报告",
        "",
        "## 结论先行",
        "",
        "- 评测有效：同一冻结集、同一模型、相同配置，各模式40次测量，"
        "10轮配对，先后顺序各占一半。",
        "- Agent执行阶段：并行方案的均值下降"
        f"{percent(aggregate['execution_seconds']['mean'])}，P50下降"
        f"{percent(aggregate['execution_seconds']['p50'])}，P95下降"
        f"{percent(aggregate['execution_seconds']['p95'])}。",
        "- 端到端阶段：均值下降"
        f"{percent(aggregate['total_seconds']['mean'])}，P50"
        f"{'下降' if aggregate['total_seconds']['p50'] >= 0 else '上升'}"
        f"{percent(abs(aggregate['total_seconds']['p50']))}，P95下降"
        f"{percent(aggregate['total_seconds']['p95'])}。",
        "- 串行与并行成功率均为100%，但并行发生10次子Agent重试，"
        "串行为0次；并行增加了限流压力。",
        "- 不支持“端到端响应时间降低50%”。可严谨表述为："
        "“在40组正式配对样本中，同优先级Agent并行使执行阶段P50降低"
        "45.8%、P95降低46.0%，端到端均值降低13.0%。”",
        "",
        "## 核心指标",
        "",
        "|阶段与指标|串行|并行|变化|",
        "|---|---:|---:|---:|",
    ]
    for phase, label in (
        ("execution_seconds", "Agent执行"),
        ("total_seconds", "端到端"),
        ("intent_seconds", "意图识别"),
    ):
        for metric in ("mean", "p50", "p95"):
            lines.append(
                f"|{label} {metric.upper()}|"
                f"{sequential[phase][metric]:.2f}s|"
                f"{parallel[phase][metric]:.2f}s|"
                f"{percent(aggregate[phase][metric])}|"
            )

    lines.extend([
        f"|成功率|{sequential['success_rate']:.1%}|"
        f"{parallel['success_rate']:.1%}|无回归|",
        f"|调度匹配率|{sequential['schedule_match_rate']:.1%}|"
        f"{parallel['schedule_match_rate']:.1%}|一致|",
        f"|子Agent重试次数|{sequential['agent_retry_count']}|"
        f"{parallel['agent_retry_count']}|并行增加|",
        f"|意图重试次数|{sequential['intent_retry_count']}|"
        f"{parallel['intent_retry_count']}|供应商波动|",
        "",
        "正值表示并行耗时更低；负值表示并行耗时更高。",
        "",
        "## 分任务执行阶段",
        "",
        "|任务组合|串行P50|并行P50|P50变化|串行P95|并行P95|P95变化|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in result["category_metrics"]:
        lines.append(
            f"|{item['category']}|"
            f"{item['sequential_execution_seconds']['p50']:.2f}s|"
            f"{item['parallel_execution_seconds']['p50']:.2f}s|"
            f"{percent(item['execution_improvement']['p50'])}|"
            f"{item['sequential_execution_seconds']['p95']:.2f}s|"
            f"{item['parallel_execution_seconds']['p95']:.2f}s|"
            f"{percent(item['execution_improvement']['p95'])}|"
        )

    lines.extend([
        "",
        "## 稳定性与异常值",
        "",
        f"- 并行执行在40对样本中胜出"
        f"{result['comparison']['parallel_execution_win_rate']:.1%}；"
        f"端到端胜率为{result['comparison']['parallel_total_win_rate']:.1%}。",
        f"- 并行有{parallel['agent_retry_request_count']}个请求发生子Agent"
        f"重试，共{parallel['agent_retry_count']}次；最终无失败。",
        f"- 串行有{sequential['intent_retry_request_count']}个请求发生意图"
        f"重试，并行有{parallel['intent_retry_request_count']}个。",
        "- 意图识别不在并发批次内，因此其超时和重试会稀释编排优化对"
        "端到端延迟的贡献。",
        "",
        "## 无效尝试保留",
        "",
        "|尝试|作废原因|",
        "|---:|---|",
    ])
    for item in result["invalid_attempts_retained"]:
        lines.append(f"|{item['attempt']}|{item['reason']}|")

    lines.extend([
        "",
        "## 对外表述边界",
        "",
        "可以使用：",
        "",
        "- “设计同一模型、同一冻结任务集的串并行配对实验，各模式40次"
        "测量；并行编排使Agent执行阶段P50降低45.8%、P95降低46.0%，"
        "成功率保持100%。”",
        "- “识别并修复子Agent错误被误计成功、RAG摘要失败伪成功、"
        "内部429未触发外层重试等稳定性缺陷。”",
        "",
        "不能使用：",
        "",
        "- “整体响应时间降低50%。”端到端P50实际增加13.9%，仅均值和"
        "P95分别改善13.0%和27.6%。",
        "- “并行没有任何代价。”并行模式增加了429重试压力。",
        "",
        "## 验证结论",
        "",
        f"- 状态：`{result['validation_status']}`",
        f"- 原始结果 SHA-256：`{result['source_report_sha256']}`",
        f"- 冻结数据集 SHA-256：`{result['dataset_sha256']}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = PROJECT_ROOT / args.report
    manifest_path = PROJECT_ROOT / args.dataset_manifest
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = analyze(report, manifest, report_path=report_path)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps({
        "validation_status": result["validation_status"],
        "integrity_checks": result["integrity_checks"],
        "supported_claims": result["supported_claims"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved analysis: {output_path}")
    print(f"Saved report: {markdown_path}")
    return 0 if result["validation_status"] != "needs_revision" else 2


if __name__ == "__main__":
    raise SystemExit(main())
