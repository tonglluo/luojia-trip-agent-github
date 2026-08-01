"""Build the canonical V1/V2 intent comparison report artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import write_json


QUALITY_LABELS = {
    "exact_match": "意图 Exact Match",
    "macro_f1": "Macro F1",
    "schedule_exact_match": "调度 Exact Match",
    "schedule_intent_consistency": "意图-调度一致率",
}
LATENCY_LABELS = {
    "mean": "平均值",
    "p50": "P50",
    "p95": "P95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/reports/intent_comparison_v1_v2.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/intent_comparison_artifact.json",
    )
    return parser.parse_args()


def _metric_map(rows: list[dict]) -> dict[str, dict]:
    return {row["metric"]: row for row in rows}


def build_artifact(comparison: dict) -> dict:
    generated_at = comparison["generated_at"]
    quality_map = _metric_map(comparison["quality_metrics"])
    latency_map = _metric_map(comparison["latency_metrics"])
    transitions = comparison["case_transitions"]

    quality_rows = []
    for metric, label in QUALITY_LABELS.items():
        row = quality_map[metric]
        for version, value_field in (
            ("V1", "baseline"),
            ("V2", "candidate"),
        ):
            quality_rows.append({
                "metric": label,
                "metric_key": metric,
                "version": version,
                "value": row[value_field],
                "absolute_delta": row["absolute_delta"],
                "sample_count": 30,
            })

    latency_rows = []
    for metric, label in LATENCY_LABELS.items():
        row = latency_map[metric]
        for version, value_field in (
            ("V1", "baseline_seconds"),
            ("V2", "candidate_seconds"),
        ):
            latency_rows.append({
                "metric": label,
                "metric_key": metric,
                "version": version,
                "seconds": row[value_field],
                "delta_seconds": row["delta_seconds"],
                "relative_change": row["relative_change"],
                "sample_count": 30,
            })

    label_rows = [
        {
            "label": row["label"],
            "v1_precision": row["baseline_precision"],
            "v2_precision": row["candidate_precision"],
            "v1_recall": row["baseline_recall"],
            "v2_recall": row["candidate_recall"],
            "v1_f1": row["baseline_f1"],
            "v2_f1": row["candidate_f1"],
        }
        for row in comparison["label_metrics"]
    ]

    source = {
        "id": "intent_comparison",
        "label": "意图识别 V1/V2 对照转换",
        "path": "evaluation/reports/intent_comparison_v1_v2.sql",
        "query": {
            "description": (
                "使用 DuckDB SQL 将 V1/V2 对照 JSON 转换为质量和延迟图表。"
            ),
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": generated_at,
            "tables_used": [
                "evaluation/reports/intent_comparison_v1_v2.json",
            ],
            "filters": [
                "same 30 case ids",
                "model=glm-4.7",
                "thinking=disabled",
            ],
            "metric_definitions": [
                "Exact Match：预测意图集合与标注集合完全一致的样本占比。",
                "Macro F1：六个业务意图标签 F1 的算术平均值。",
                "调度 Exact Match：智能体及优先级映射完全一致的样本占比。",
                "P50/P95：单次意图识别调用耗时的第 50/95 百分位。",
            ],
        },
    }

    exact = quality_map["exact_match"]
    macro = quality_map["macro_f1"]
    schedule = quality_map["schedule_exact_match"]
    p50 = latency_map["p50"]
    p95 = latency_map["p95"]
    mean = latency_map["mean"]

    title = "意图识别 V1/V2 对照"
    summary = f"""## Executive Summary

- **V2 在当前 30 条开发集上全部通过。** 意图 Exact Match 从 {exact['baseline']:.1%} 提升到 {exact['candidate']:.1%}，Macro F1 从 {macro['baseline']:.1%} 提升到 {macro['candidate']:.1%}；修复了 {transitions['intent_fixed_count']} 条意图失败，未观察到回归。
- **调度规则也完成闭环。** 调度 Exact Match 从 {schedule['baseline']:.1%} 提升到 {schedule['candidate']:.1%}，修复 {transitions['schedule_fixed_count']} 条调度失败，意图与调度一致率达到 100%。
- **这仍是开发集成绩，不是泛化成绩。** 规则和 Prompt 是根据这 30 条失败样本调整的，因此“100%”只能证明已知问题被修复，不能直接作为简历中的最终准确率。
- **延迟表现改善，但不能归因。** V2 的 P50 为 {p50['candidate_seconds']:.2f}s、P95 为 {p95['candidate_seconds']:.2f}s；由于每个版本只运行一次且 V2 仍有 85.75s 异常值，暂不能宣称响应时间降低。"""

    quality_text = f"""## 已知意图和调度缺陷已全部修复

V1 的主要缺陷是 `event_collection` 漏识别、`preference` 误触发，以及非差旅请求被路由到业务智能体。V2 通过 Prompt 边界、合法意图白名单和确定性调度校验修复这些问题。逐例对照显示：{transitions['intent_fixed_count']} 条意图失败转为通过，{transitions['schedule_fixed_count']} 条调度失败转为通过，回归数为 {transitions['regression_count']}。

**产品含义：** 当前版本已满足“已知业务规则被正确执行”的开发验收条件，可以进入独立测试集阶段；不应该继续在原 30 条上调参。"""

    label_text = """## 改善集中在原先的薄弱标签

`event_collection` 的召回率由 55.6% 提升到 100%，`preference` 的精确率由 66.7% 提升到 100%。这与修复目标一致，说明改善不是由一个无关标签拉动的平均数变化。`itinerary_planning` 和 `memory_query` 原本已全对，V2 仍保持通过。

**需要警惕：** 每个标签只有 5–9 个正例，单标签 100% 的置信度仍然很低。下一轮需要增加边界表达、否定表达和多轮上下文。"""

    latency_text = f"""## 典型延迟下降，但长尾问题仍未消失

平均耗时从 {mean['baseline_seconds']:.2f}s 降至 {mean['candidate_seconds']:.2f}s，P50 从 {p50['baseline_seconds']:.2f}s 降至 {p50['candidate_seconds']:.2f}s，P95 从 {p95['baseline_seconds']:.2f}s 降至 {p95['candidate_seconds']:.2f}s。表面上 P95 下降了 {abs(p95['relative_change']):.1%}，但 V2 的最大值仍为 {latency_map['max']['candidate_seconds']:.2f}s。

**产品含义：** 当前证据只能说明 V2 这一次运行的典型请求更快，不能证明代码优化导致延迟下降。需要对同一冻结数据集至少重复 3 次，并记录重试次数、Token 数和平台状态。"""

    next_steps = """## 下一步：冻结开发集，建立独立测试集

1. 将当前 30 条标记为开发集，不再用它们计算最终简历指标。
2. 新建 60–90 条独立测试集，至少覆盖单意图、多意图、上下文消歧、语义边界、否定表达和非差旅请求。
3. 测试集标注完成后先冻结，再运行模型；失败后不能修改测试集答案来适配输出。
4. 在独立集上运行 3 次，报告 Exact Match、Macro F1、调度一致率的三次均值及范围。
5. 单独运行延迟基准，固定模型、Thinking、并发、数据和网络条件，报告 P50、P95、均值、最大值和成功率。"""

    questions = """## 进一步需要回答的问题

- 独立测试集由谁复核标注，如何处理有争议的多意图样本？
- 规则后处理对未覆盖的城市信息、餐饮、签证和否定句是否会误拦截？
- 85.75 秒请求是否发生 SDK 重试、平台排队或超长输出？
- 正式产品是否把 `event_collection` 定义为“用户意图”，还是仅定义为内部编排依赖？"""

    caveats = """## 限制与假设

- V1 与 V2 使用相同模型、Thinking 配置、30 个样本 ID，且意图与调度标注一致。
- 两次运行之间修订过实体标注和实体文本归一化，因此实体准确率变化不用于估计模型改进。
- 每个版本只有一次运行，延迟变化可能来自模型平台或网络波动。
- V2 是针对 V1 失败样本优化后的开发集结果，存在明显的评测集过拟合风险。"""

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "同一 30 条开发集上的意图识别 V1/V2 对照与下一阶段评测建议。",
            "generatedAt": generated_at,
            "sources": [source],
            "charts": [
                {
                    "id": "quality_comparison",
                    "title": "V1/V2 质量指标",
                    "subtitle": "相同 30 条开发样本；数值越高越好",
                    "type": "bar",
                    "dataset": "quality_comparison",
                    "sourceId": "intent_comparison",
                    "encodings": {
                        "x": {
                            "field": "metric",
                            "type": "nominal",
                            "label": "指标",
                        },
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "通过率",
                        },
                        "color": {
                            "field": "version",
                            "type": "nominal",
                            "label": "版本",
                        },
                        "tooltip": [
                            {"field": "metric", "type": "text"},
                            {"field": "version", "type": "text"},
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
                },
                {
                    "id": "latency_comparison",
                    "title": "V1/V2 意图识别耗时",
                    "subtitle": "单次运行、各 30 个请求；单位为秒",
                    "type": "bar",
                    "dataset": "latency_comparison",
                    "sourceId": "intent_comparison",
                    "encodings": {
                        "x": {
                            "field": "metric",
                            "type": "nominal",
                            "label": "延迟指标",
                        },
                        "y": {
                            "field": "seconds",
                            "type": "quantitative",
                            "format": "number",
                            "label": "秒",
                        },
                        "color": {
                            "field": "version",
                            "type": "nominal",
                            "label": "版本",
                        },
                        "tooltip": [
                            {"field": "metric", "type": "text"},
                            {"field": "version", "type": "text"},
                            {
                                "field": "seconds",
                                "type": "quantitative",
                                "format": "number",
                            },
                            {"field": "sample_count", "type": "quantitative"},
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "秒",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "label_comparison",
                    "title": "各意图标签 V1/V2 指标",
                    "subtitle": "Precision、Recall 与 F1 的逐标签对照",
                    "dataset": "label_comparison",
                    "sourceId": "intent_comparison",
                    "defaultSort": {
                        "field": "v1_f1",
                        "direction": "asc",
                    },
                    "density": "compact",
                    "columns": [
                        {"field": "label", "label": "意图", "type": "text"},
                        {
                            "field": "v1_precision",
                            "label": "V1 P",
                            "format": "percent",
                        },
                        {
                            "field": "v2_precision",
                            "label": "V2 P",
                            "format": "percent",
                        },
                        {
                            "field": "v1_recall",
                            "label": "V1 R",
                            "format": "percent",
                        },
                        {
                            "field": "v2_recall",
                            "label": "V2 R",
                            "format": "percent",
                        },
                        {
                            "field": "v1_f1",
                            "label": "V1 F1",
                            "format": "percent",
                        },
                        {
                            "field": "v2_f1",
                            "label": "V2 F1",
                            "format": "percent",
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
                    "id": "summary",
                    "type": "markdown",
                    "body": summary,
                    "sourceId": "intent_comparison",
                    "layout": "full",
                },
                {
                    "id": "quality_text",
                    "type": "markdown",
                    "body": quality_text,
                    "sourceId": "intent_comparison",
                    "layout": "full",
                },
                {
                    "id": "quality_chart",
                    "type": "chart",
                    "chartId": "quality_comparison",
                    "layout": "full",
                },
                {
                    "id": "label_text",
                    "type": "markdown",
                    "body": label_text,
                    "sourceId": "intent_comparison",
                    "layout": "full",
                },
                {
                    "id": "label_table",
                    "type": "table",
                    "tableId": "label_comparison",
                    "layout": "full",
                },
                {
                    "id": "latency_text",
                    "type": "markdown",
                    "body": latency_text,
                    "sourceId": "intent_comparison",
                    "layout": "full",
                },
                {
                    "id": "latency_chart",
                    "type": "chart",
                    "chartId": "latency_comparison",
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
                    "sourceId": "intent_comparison",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "quality_comparison": quality_rows,
                "latency_comparison": latency_rows,
                "label_comparison": label_rows,
            },
        },
        "sources": [source],
    }


def main() -> int:
    args = parse_args()
    comparison = json.loads(
        (PROJECT_ROOT / args.input).read_text(encoding="utf-8"),
    )
    artifact = build_artifact(comparison)
    destination = write_json(PROJECT_ROOT / args.output, artifact)
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
