"""Create a threshold check and cross-run error analysis for intent runs."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import utc_now_iso, write_json


THRESHOLDS = {
    "exact_match": 0.90,
    "macro_f1": 0.95,
    "schedule_exact_match": 0.95,
    "entity_field_accuracy": 0.90,
    "error_rate": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="evaluation/reports/intent_holdout_v2_repeated",
    )
    parser.add_argument(
        "--output",
        default=(
            "evaluation/reports/intent_holdout_v2_repeated/"
            "error_analysis.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        default=(
            "evaluation/reports/intent_holdout_v2_repeated/"
            "final_report.md"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def analyze(input_dir: Path) -> dict[str, Any]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Repeated-run summary not found: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("valid_run_count", 0) < 3:
        raise SystemExit("At least three valid runs are required")

    valid_attempts = [
        attempt for attempt in summary["attempts"] if attempt["valid"]
    ]
    reports = [
        load_json(PROJECT_ROOT / attempt["result"])
        for attempt in valid_attempts
    ]
    expected_hash = summary["dataset_sha256"]
    expected_count = summary["sample_count_per_run"]
    if any(
        report.get("dataset_sha256") != expected_hash
        or report.get("sample_count") != expected_count
        or report.get("metrics", {}).get("error_rate") != 0
        for report in reports
    ):
        raise SystemExit("One or more runs fail hash/count/error validation")

    label_names = sorted({
        label
        for report in reports
        for label in report["metrics"]["per_label"]
    })
    per_label: dict[str, dict[str, dict[str, float]]] = {}
    for label in label_names:
        per_label[label] = {}
        for metric in ("precision", "recall", "f1"):
            values = [
                float(report["metrics"]["per_label"][label][metric])
                for report in reports
            ]
            per_label[label][metric] = summarize(values)

    cases: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "category": None,
            "intent_fail_runs": [],
            "schedule_fail_runs": [],
            "entity_fail_runs": [],
            "failed_entity_fields": Counter(),
            "predicted_intent_sets": Counter(),
        },
    )
    for run_number, report in enumerate(reports, 1):
        for row in report["rows"]:
            case = cases[row["id"]]
            case["category"] = row.get("category")
            predicted_key = ", ".join(row["predicted_intents"]) or "(empty)"
            case["predicted_intent_sets"][predicted_key] += 1
            if not row["intent_exact"]:
                case["intent_fail_runs"].append(run_number)
            if not row["schedule_exact"]:
                case["schedule_fail_runs"].append(run_number)
            failed_fields = [
                field
                for field, passed in row.get("entity_checks", {}).items()
                if not passed
            ]
            if failed_fields:
                case["entity_fail_runs"].append(run_number)
                case["failed_entity_fields"].update(failed_fields)

    run_count = len(reports)
    case_rows = []
    for case_id, case in sorted(cases.items()):
        any_fail_runs = sorted(set(
            case["intent_fail_runs"]
            + case["schedule_fail_runs"]
            + case["entity_fail_runs"]
        ))
        if not any_fail_runs:
            continue
        case_rows.append({
            "id": case_id,
            "category": case["category"],
            "intent_fail_runs": case["intent_fail_runs"],
            "schedule_fail_runs": case["schedule_fail_runs"],
            "entity_fail_runs": case["entity_fail_runs"],
            "failed_entity_fields": dict(
                case["failed_entity_fields"].most_common(),
            ),
            "predicted_intent_sets": dict(
                case["predicted_intent_sets"].most_common(),
            ),
            "failure_type": (
                "common"
                if len(any_fail_runs) == run_count
                else "variable"
            ),
        })

    common_failures = [
        row for row in case_rows if row["failure_type"] == "common"
    ]
    variable_failures = [
        row for row in case_rows if row["failure_type"] == "variable"
    ]
    threshold_checks = {}
    for metric, threshold in THRESHOLDS.items():
        actual = float(summary["metrics"][metric]["mean"])
        passed = actual <= threshold if metric == "error_rate" else actual >= threshold
        threshold_checks[metric] = {
            "actual_mean": actual,
            "threshold": threshold,
            "passed": passed,
        }

    return {
        "generated_at": utc_now_iso(),
        "dataset": summary["dataset"],
        "dataset_sha256": expected_hash,
        "model": summary["model"],
        "thinking": summary["thinking"],
        "valid_run_count": run_count,
        "sample_count_per_run": expected_count,
        "threshold_checks": threshold_checks,
        "all_quality_thresholds_passed": all(
            item["passed"] for item in threshold_checks.values()
        ),
        "metrics": summary["metrics"],
        "per_label": per_label,
        "failure_summary": {
            "cases_with_any_failure": len(case_rows),
            "common_failure_count": len(common_failures),
            "variable_failure_count": len(variable_failures),
        },
        "common_failures": common_failures,
        "variable_failures": variable_failures,
        "latency_all_valid_requests": summary[
            "latency_all_valid_requests"
        ],
        "attempts": summary["attempts"],
        "method_note": (
            "Common failures occur in all valid runs; variable failures occur "
            "in only some runs. Frozen annotations were not changed."
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# 意图识别 V2 三轮正式评测报告",
        "",
        f"- 冻结集：`{result['dataset']}`",
        f"- SHA-256：`{result['dataset_sha256']}`",
        f"- 模型 / Thinking：{result['model']} / {result['thinking']}",
        f"- 有效运行：{result['valid_run_count']} 次，每次 "
        f"{result['sample_count_per_run']} 条",
        f"- 总体门槛："
        f"{'通过' if result['all_quality_thresholds_passed'] else '未通过'}",
        "",
        "## 核心指标",
        "",
        "|指标|均值|最小|最大|门槛|结论|",
        "|---|---:|---:|---:|---:|---|",
    ]
    labels = {
        "exact_match": "意图集合完全匹配率",
        "macro_f1": "Macro F1",
        "schedule_exact_match": "调度完全匹配率",
        "entity_field_accuracy": "实体字段准确率",
        "error_rate": "错误率",
    }
    for metric, display_name in labels.items():
        values = result["metrics"][metric]
        check = result["threshold_checks"][metric]
        operator = "≤" if metric == "error_rate" else "≥"
        lines.append(
            f"|{display_name}|{values['mean']:.2%}|{values['min']:.2%}|"
            f"{values['max']:.2%}|{operator}{check['threshold']:.0%}|"
            f"{'通过' if check['passed'] else '未通过'}|"
        )

    lines.extend([
        "",
        "## 逐类 F1",
        "",
        "|意图|F1均值|最小|最大|",
        "|---|---:|---:|---:|",
    ])
    for label, metrics in result["per_label"].items():
        f1 = metrics["f1"]
        lines.append(
            f"|{label}|{f1['mean']:.2%}|{f1['min']:.2%}|"
            f"{f1['max']:.2%}|"
        )

    failure_summary = result["failure_summary"]
    lines.extend([
        "",
        "## 错误归因",
        "",
        f"- 至少一次失败的样本：{failure_summary['cases_with_any_failure']}",
        f"- 三轮共同失败：{failure_summary['common_failure_count']}",
        f"- 随机波动失败：{failure_summary['variable_failure_count']}",
        "",
        "|ID|类别|类型|意图失败轮次|调度失败轮次|实体失败轮次|实体字段|",
        "|---|---|---|---|---|---|---|",
    ])
    for row in result["common_failures"] + result["variable_failures"]:
        fields = "、".join(row["failed_entity_fields"]) or "—"
        lines.append(
            f"|{row['id']}|{row['category']}|"
            f"{'共同' if row['failure_type'] == 'common' else '波动'}|"
            f"{row['intent_fail_runs'] or '—'}|"
            f"{row['schedule_fail_runs'] or '—'}|"
            f"{row['entity_fail_runs'] or '—'}|{fields}|"
        )
    if not result["common_failures"] and not result["variable_failures"]:
        lines.append("|—|—|无失败|—|—|—|—|")

    latency = result["latency_all_valid_requests"]
    lines.extend([
        "",
        "## 运行说明",
        "",
        f"- 有效请求数：{latency['request_count']}",
        f"- 延迟均值 / P50 / P95：{latency['mean_seconds']:.2f}s / "
        f"{latency['p50_seconds']:.2f}s / {latency['p95_seconds']:.2f}s",
        "- 所有尝试均保留；只有数据集哈希一致、样本完整且错误率为 0 "
        "的运行进入聚合。",
        "- 本报告只证明当前冻结测试集上的表现，不等同于线上真实流量准确率。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = analyze(PROJECT_ROOT / args.input_dir)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps(result["threshold_checks"], ensure_ascii=False, indent=2))
    print(f"Saved analysis: {output_path}")
    print(f"Saved report: {markdown_path}")
    return 0 if result["all_quality_thresholds_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
