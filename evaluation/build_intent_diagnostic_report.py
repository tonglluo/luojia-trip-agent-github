"""Build a canonical portable-report artifact for an intent baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/reports/intent_diagnostics.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_diagnostic_artifact.json",
    )
    return parser.parse_args()


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_artifact(diagnostic: dict) -> dict:
    generated_at = diagnostic["generated_at"]
    metrics = diagnostic["headline_metrics"]
    per_label = metrics["per_label"]

    quality_rows = []
    for label, values in per_label.items():
        quality_rows.extend([
            {
                "intent": label,
                "metric": "Precision",
                "value": values["precision"],
                "sample_count": diagnostic["sample_count"],
            },
            {
                "intent": label,
                "metric": "Recall",
                "value": values["recall"],
                "sample_count": diagnostic["sample_count"],
            },
        ])

    category_rows = [
        {
            "category": category,
            "count": values["count"],
            "intent_exact_match": values["intent_exact_match"],
            "schedule_exact_match": values["schedule_exact_match"],
            "entity_field_accuracy": values["entity_field_accuracy"],
        }
        for category, values in diagnostic["category_metrics"].items()
    ]

    failure_rows = []
    for failure in diagnostic["failures"]:
        issues = []
        if failure["missing_intents"]:
            issues.append(
                "缺失意图: " + ", ".join(failure["missing_intents"]),
            )
        if failure["extra_intents"]:
            issues.append(
                "多余意图: " + ", ".join(failure["extra_intents"]),
            )
        if failure["missing_scheduled_agents"]:
            issues.append(
                "缺失调度: " + ", ".join(
                    failure["missing_scheduled_agents"],
                ),
            )
        if failure["extra_scheduled_agents"]:
            issues.append(
                "多余调度: " + ", ".join(
                    failure["extra_scheduled_agents"],
                ),
            )
        if failure["entity_failures"]:
            issues.append(
                "实体字段: " + ", ".join(failure["entity_failures"]),
            )
        failure_rows.append({
            "id": failure["id"],
            "category": failure["category"],
            "issue": "；".join(issues) or "运行错误",
            "latency_seconds": failure["latency_seconds"],
        })

    source = {
        "id": "intent_diagnostic",
        "label": "意图识别基线诊断转换",
        "path": "evaluation/reports/intent_diagnostic.sql",
        "query": {
            "description": (
                "使用 DuckDB SQL 将 30 条意图评测诊断 JSON 转换为图表和明细表。"
            ),
            "language": "sql",
            "engine": "DuckDB",
            "executed_at": generated_at,
            "tables_used": [
                "evaluation/reports/intent_diagnostics.json",
            ],
            "filters": [
                "dataset=evaluation/datasets/intent_eval.sample.jsonl",
                "thinking=disabled",
                "sample_count=30",
            ],
            "metric_definitions": [
                "Exact Match：一条样本的预测意图集合与标注意图集合完全相同。",
                "Macro F1：六个受支持意图标签 F1 的算术平均值。",
                "Schedule Exact Match：智能体及其优先级映射与标注完全相同。",
                "Entity Field Accuracy：通过字段数除以所有被标注字段数。",
                "P50/P95：30 次成功意图识别调用耗时的第 50/95 百分位。",
            ],
        },
    }

    title = "差旅助手意图识别基线诊断"
    executive_summary = f"""## Executive Summary

- **当前基线还不能写成“意图识别准确率 90%+”。** 30 条样本的 Exact Match 为 {_percent(metrics['exact_match'])}；Macro F1 为 {_percent(metrics['macro_f1'])}，两者口径不同，简历不能混用。
- **主要错误集中在两条路由规则。** `event_collection` 缺失 4 次；`preference` 多报 3 次。前者拉低规划类多意图召回，后者让单一行程请求产生不必要的偏好调用。
- **调度稳定性与尾延迟仍需优化。** Schedule Exact Match 为 {_percent(metrics['schedule_exact_match'])}；意图识别耗时 P50 为 {diagnostic['latency_seconds']['p50']:.2f}s、P95 为 {diagnostic['latency_seconds']['p95']:.2f}s，最慢请求为 {diagnostic['latency_seconds']['max']:.2f}s。
- **本轮已完成规则修复，但尚未重新调用模型验证。** 下一步应用同一数据集复测，并保留修改前后两份结果进行对比。"""

    findings = f"""## 错误集中在规划前置步骤与偏好误触发

`itinerary_planning`、`memory_query` 在本次运行中各自达到 100% Precision / Recall，但不能据此宣称泛化达到 100%，因为对应正例数量仅为 9 条和 6 条。最弱标签是 `event_collection`：Precision 100%，Recall 55.6%，说明它没有乱报，但在 4 条多意图规划请求中被漏写进 `intents`。

`preference` 的 Recall 为 100%，Precision 为 66.7%。3 条纯行程请求被额外识别为偏好意图，原因是模型把“读取历史偏好可能有助于规划”误当作用户的偏好声明。产品规则应坚持：只有用户明确声明或修改偏好时，才触发该智能体。"""

    latency = f"""## 尾延迟比平均值更值得优先处理

本次意图识别平均耗时为 {diagnostic['latency_seconds']['mean']:.2f}s，但 P95 达到 {diagnostic['latency_seconds']['p95']:.2f}s，是 P50（{diagnostic['latency_seconds']['p50']:.2f}s）的约 {diagnostic['latency_seconds']['p95'] / diagnostic['latency_seconds']['p50']:.1f} 倍。最慢的 3 条分别为 90.62s、64.72s 和 47.73s，且都返回成功，说明问题更像模型服务抖动、重试或长生成，而不是代码异常。

下一轮延迟基准应单独记录首 Token 时间、重试次数、输入/输出 Token 数和 HTTP 状态；否则只能确认“慢”，无法区分网络、平台排队和 Prompt 长度。"""

    next_steps = """## 建议的下一步

1. 使用同一份 30 条数据集复跑一次，验证本轮规则修复是否把 `event_collection` 漏识别和 `preference` 误触发消除。
2. 若 Exact Match 达到目标，再复跑 2 次，报告三次均值、最小值和最大值，避免把单次随机结果当成稳定表现。
3. 人工复核失败条目和修订后的实体标注，再扩充到 50–100 条；每个核心意图至少保留 10 条，并加入相似边界、否定表达和多轮上下文。
4. 单独执行优化前/后的延迟基准，固定模型、Prompt、数据、`thinking=disabled` 和并发参数，至少报告 P50、P95、均值和成功率。
5. 只有完成复测并冻结测试集后，才把可复现指标写入 README 或简历。"""

    questions = """## 进一步需要验证的问题

- GLM 平台在 40–90 秒请求上是否发生了 SDK 自动重试或服务端排队？
- `information_query` 的业务边界是否明确限定为差旅相关公开信息？
- 相对日期实体是否需要单独按“日期解析正确率”评测，而不是混入通用实体准确率？
- 未来真实产品是否把 `event_collection` 视为用户意图，还是只视为编排依赖？当前测试集采用前一种口径。"""

    caveats = """## 限制与假设

- 这是单次、30 条样本的开发基线，不是最终评测，也不是可直接用于简历的结论。
- 本轮结果来自 GLM-4.7、`thinking=disabled`；模型服务状态和随机性可能影响结果。
- 基线运行后修正了“杭州东站到西湖”的目的地标注，以及中文/数字时长等价口径；修复后的实体指标必须通过下一次完整运行重新计算。
- 报告中的诊断解释来自逐例预测结果；延迟成因仍是待验证假设。"""

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "30 条意图识别评测的基线结果、错误归因和复测计划。",
            "generatedAt": generated_at,
            "sources": [source],
            "charts": [
                {
                    "id": "intent_precision_recall",
                    "title": "各意图 Precision 与 Recall",
                    "subtitle": "30 条开发样本基线；数值越高越好",
                    "type": "bar",
                    "dataset": "intent_quality",
                    "sourceId": "intent_diagnostic",
                    "encodings": {
                        "x": {
                            "field": "intent",
                            "type": "nominal",
                            "label": "意图",
                        },
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "指标值",
                        },
                        "color": {
                            "field": "metric",
                            "type": "nominal",
                            "label": "指标",
                        },
                        "tooltip": [
                            {"field": "intent", "type": "text"},
                            {"field": "metric", "type": "text"},
                            {
                                "field": "value",
                                "type": "quantitative",
                                "format": "percent",
                            },
                            {"field": "sample_count", "type": "quantitative"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                    "maxRows": 20,
                },
            ],
            "tables": [
                {
                    "id": "category_performance",
                    "title": "各场景评测结果",
                    "subtitle": "Exact Match 与实体字段准确率",
                    "dataset": "category_performance",
                    "sourceId": "intent_diagnostic",
                    "defaultSort": {
                        "field": "intent_exact_match",
                        "direction": "asc",
                    },
                    "density": "compact",
                    "columns": [
                        {"field": "category", "label": "场景", "type": "text"},
                        {"field": "count", "label": "样本数", "type": "number"},
                        {
                            "field": "intent_exact_match",
                            "label": "意图 EM",
                            "format": "percent",
                        },
                        {
                            "field": "schedule_exact_match",
                            "label": "调度 EM",
                            "format": "percent",
                        },
                        {
                            "field": "entity_field_accuracy",
                            "label": "实体准确率",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "failure_details",
                    "title": "失败样本明细",
                    "subtitle": "按调用耗时降序",
                    "dataset": "failure_details",
                    "sourceId": "intent_diagnostic",
                    "defaultSort": {
                        "field": "latency_seconds",
                        "direction": "desc",
                    },
                    "density": "compact",
                    "columns": [
                        {"field": "id", "label": "ID", "type": "text"},
                        {"field": "category", "label": "场景", "type": "text"},
                        {"field": "issue", "label": "问题", "type": "text"},
                        {
                            "field": "latency_seconds",
                            "label": "耗时（秒）",
                            "type": "number",
                        },
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {title}",
                    "layout": "full",
                },
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "body": executive_summary,
                    "sourceId": "intent_diagnostic",
                    "layout": "full",
                },
                {
                    "id": "findings",
                    "type": "markdown",
                    "body": findings,
                    "sourceId": "intent_diagnostic",
                    "layout": "full",
                },
                {
                    "id": "quality_chart",
                    "type": "chart",
                    "chartId": "intent_precision_recall",
                    "layout": "full",
                },
                {
                    "id": "category_table",
                    "type": "table",
                    "tableId": "category_performance",
                    "layout": "full",
                },
                {
                    "id": "latency",
                    "type": "markdown",
                    "body": latency,
                    "sourceId": "intent_diagnostic",
                    "layout": "full",
                },
                {
                    "id": "failure_table",
                    "type": "table",
                    "tableId": "failure_details",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": next_steps,
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": questions,
                    "layout": "full",
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "body": caveats,
                    "sourceId": "intent_diagnostic",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "intent_quality": quality_rows,
                "category_performance": category_rows,
                "failure_details": failure_rows,
            },
        },
        "sources": [source],
    }


def main() -> int:
    args = parse_args()
    diagnostic = json.loads(
        (PROJECT_ROOT / args.input).read_text(encoding="utf-8"),
    )
    artifact = build_artifact(diagnostic)
    destination = write_json(PROJECT_ROOT / args.output, artifact)
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
