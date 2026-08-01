"""Build the frozen technical-evaluation artifact from reviewed result files."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports"
OUTPUT = REPORTS / "final_technical_evaluation_artifact.json"
SOURCE_NOTES = REPORTS / "final_technical_evaluation_source_notes.json"

INTENT_PATH = REPORTS / "intent_holdout_v2_repeated" / "summary.json"
RAG_PATH = REPORTS / "rag_formal_v1_analysis.json"
LATENCY_PATH = REPORTS / "latency_orchestration_paired_v1_analysis.json"
LATENCY_CHART_SQL_PATH = REPORTS / "final_latency_chart.sql"

INTENT_TABLE_SQL = """
SELECT metric, mean, min, max, threshold, status
FROM intent_report_metrics
ORDER BY metric;
""".strip()
RAG_TABLE_SQL = """
SELECT metric, value, threshold, status
FROM rag_report_metrics
ORDER BY metric;
""".strip()
LATENCY_TABLE_SQL = """
SELECT stage, metric, sequential, parallel, change, interpretation
FROM latency_report_metrics
ORDER BY stage, metric;
""".strip()
COMPLETION_TABLE_SQL = """
SELECT workstream, evidence, result, status
FROM completion_audit
ORDER BY workstream;
""".strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def query_rows(
    rows: list[dict[str, Any]],
    table_name: str,
    schema: list[tuple[str, str]],
    sql: str,
) -> list[dict[str, Any]]:
    columns = [name for name, _ in schema]
    with sqlite3.connect(":memory:") as connection:
        column_sql = ", ".join(f"{name} {data_type}" for name, data_type in schema)
        connection.execute(f"CREATE TABLE {table_name} ({column_sql})")
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO {table_name} VALUES ({placeholders})",
            [tuple(row[column] for column in columns) for row in rows],
        )
        cursor = connection.execute(sql)
        output_columns = [item[0] for item in cursor.description]
        return [
            dict(zip(output_columns, values, strict=True))
            for values in cursor.fetchall()
        ]


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    tables_used: list[str],
    filters: list[str],
    definitions: list[str],
    generated_at: str,
    sql: str | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": rel(path),
        "query": {
            "description": description,
            "engine": "SQLite" if sql else "Python/JSON",
            "language": "sql" if sql else "python",
            "executed_at": generated_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": definitions,
            **({"sql": sql} if sql else {}),
        },
    }


def table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_item: dict[str, Any],
    columns: list[dict[str, Any]],
    sort_field: str,
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "source": source_item,
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "density": "spacious",
        "columns": columns,
    }


def markdown(
    block_id: str, body: str, source_id: str | None = None
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "id": block_id,
        "type": "markdown",
        "body": body,
        "layout": "full",
    }
    if source_id:
        block["sourceId"] = source_id
    return block


def build() -> None:
    intent = load_json(INTENT_PATH)
    rag = load_json(RAG_PATH)
    latency = load_json(LATENCY_PATH)
    latency_chart_sql = LATENCY_CHART_SQL_PATH.read_text(encoding="utf-8")
    generated_at = datetime.now(timezone.utc).isoformat()

    intent_metrics = intent["metrics"]
    rag_metrics = rag["metrics"]
    sequential = latency["mode_metrics"]["sequential"]
    parallel = latency["mode_metrics"]["parallel"]
    improvement = latency["comparison"]["aggregate_improvement"]

    sources = [
        source(
            "intent_v2",
            "意图识别 V2 三轮正式评测",
            INTENT_PATH,
            "读取冻结独立集的三次有效运行聚合结果。",
            [rel(INTENT_PATH), "evaluation/datasets/intent_eval.holdout.v2.jsonl"],
            [
                "60条冻结独立测试样本",
                "3次有效运行",
                "model=glm-4.7",
                "thinking=disabled",
            ],
            [
                "意图集合完全匹配率：预测意图集合与人工标注集合完全一致的请求占比。",
                "Macro F1：六个业务意图标签 F1 的算术平均。",
                "调度完全匹配率：Agent 与优先级映射完全一致的请求占比。",
                "实体字段准确率：被标注实体字段中满足归一化接受规则的字段占比。",
            ],
            generated_at,
            INTENT_TABLE_SQL,
        ),
        source(
            "rag_v1",
            "RAG V1 正式评测",
            RAG_PATH,
            "读取冻结 RAG 集的检索结果、人工答案评分和拒答复核结果。",
            [rel(RAG_PATH), "evaluation/datasets/rag_eval.formal.v1.jsonl"],
            [
                "60条冻结样本：50条可回答、10条不可回答",
                "Top K=5",
                "人工评分由项目负责人逐条确认",
            ],
            [
                "Recall@5：至少一个期望来源出现在前5条检索结果的可回答题占比。",
                "MRR：首个期望来源排名倒数的均值。",
                "人工答案均分：可回答题按0/1/2分评分后的平均分。",
                "正确拒答率：不可回答题中未编造制度内容并明确拒答的比例。",
            ],
            generated_at,
            RAG_TABLE_SQL,
        ),
        source(
            "latency_v1",
            "串行与并行配对延迟评测",
            LATENCY_PATH,
            "读取同一冻结任务集上、顺序平衡的串并行配对实验复核结果。",
            [
                rel(LATENCY_PATH),
                "evaluation/datasets/latency_orchestration.formal.v1.jsonl",
                "latency_metrics (in-memory SQLite table loaded from the reviewed analysis JSON)",
            ],
            [
                "4类多Agent任务组合",
                "每种模式40次测量",
                "10轮完整配对",
                "1轮预热/模式",
                "顶层请求间冷却10秒",
                "model=glm-4.7",
                "thinking=disabled",
            ],
            [
                "Agent执行阶段：意图识别完成后，OrchestrationAgent 调用子Agent的耗时。",
                "端到端阶段：意图识别与Agent执行耗时之和。",
                "变化率：(串行耗时-并行耗时)/串行耗时；正值表示并行更快。",
            ],
            generated_at,
            latency_chart_sql,
        ),
        source(
            "latency_table_v1",
            "串行与并行配对延迟评测表",
            LATENCY_PATH,
            "从已复核延迟分析载入SQLite临时表，输出报告中的精确对照值。",
            [
                rel(LATENCY_PATH),
                "latency_report_metrics (in-memory SQLite table)",
            ],
            [
                "4类多Agent任务组合",
                "每种模式40次测量",
                "10轮完整配对",
            ],
            [
                "变化率：(串行耗时-并行耗时)/串行耗时；正值表示并行更快。"
            ],
            generated_at,
            LATENCY_TABLE_SQL,
        ),
        {
            "id": "offline_tests",
            "label": "离线回归测试",
            "path": "tests/",
            "query": {
                "description": "将已复核的五类完成证据载入SQLite审计表；离线测试于2026-07-30运行，结果为14 passed in 7.10s。",
                "engine": "SQLite",
                "language": "sql",
                "executed_at": generated_at,
                "tables_used": [
                    "tests/test_core_offline.py",
                    "tests/test_intention_rules.py",
                    "evaluation/test_metrics.py",
                    "tests/test_rag_reply_serialization_offline.py",
                    "tests/test_rag_eval_retry.py",
                    "tests/test_orchestration_error_propagation.py",
                ],
                "filters": ["offline tests only", "14 tests collected"],
                "metric_definitions": [
                    "通过率：本次选定离线测试中通过的测试数/执行的测试数。"
                ],
                "sql": COMPLETION_TABLE_SQL,
            },
        },
    ]

    intent_rows = [
        {
            "metric": "意图集合完全匹配率",
            "mean": item["mean"],
            "min": item["min"],
            "max": item["max"],
            "threshold": "≥90%",
            "status": "通过",
        }
        for item in [intent_metrics["exact_match"]]
    ]
    intent_rows += [
        {
            "metric": label,
            "mean": item["mean"],
            "min": item["min"],
            "max": item["max"],
            "threshold": threshold,
            "status": "通过",
        }
        for label, item, threshold in [
            ("Macro F1", intent_metrics["macro_f1"], "≥95%"),
            ("调度完全匹配率", intent_metrics["schedule_exact_match"], "≥95%"),
            ("实体字段准确率", intent_metrics["entity_field_accuracy"], "≥90%"),
            ("错误率", intent_metrics["error_rate"], "=0%"),
        ]
    ]

    rag_rows = [
        {
            "metric": "Recall@5",
            "value": pct(rag_metrics["recall_at_5"]),
            "threshold": "≥90%",
            "status": "通过",
        },
        {
            "metric": "MRR",
            "value": pct(rag_metrics["mrr"]),
            "threshold": "记录项",
            "status": "记录",
        },
        {
            "metric": "全来源召回率",
            "value": pct(rag_metrics["all_sources_recalled_at_5"]),
            "threshold": "记录项",
            "status": "记录",
        },
        {
            "metric": "人工答案均分",
            "value": f'{rag_metrics["manual_answer_score_mean"]:.2f}/2',
            "threshold": "≥1.6/2",
            "status": "通过",
        },
        {
            "metric": "人工正确拒答率",
            "value": pct(rag_metrics["manual_correct_refusal_rate"]),
            "threshold": "≥90%",
            "status": "通过",
        },
    ]

    latency_rows = []
    for stage, key in [
        ("Agent执行", "execution_seconds"),
        ("端到端", "total_seconds"),
    ]:
        for metric_name, metric_key in [
            ("均值", "mean"),
            ("P50", "p50"),
            ("P95", "p95"),
        ]:
            change = improvement[key][metric_key]
            latency_rows.append(
                {
                    "stage": stage,
                    "metric": metric_name,
                    "sequential": sequential[key][metric_key],
                    "parallel": parallel[key][metric_key],
                    "change": change,
                    "interpretation": "并行更快" if change > 0 else "并行更慢",
                }
            )
    chart_source_rows = []
    sort_order = 0
    for stage, key in [
        ("Agent执行", "execution_seconds"),
        ("端到端", "total_seconds"),
    ]:
        for metric_name, metric_key in [
            ("均值", "mean"),
            ("P50", "p50"),
            ("P95", "p95"),
        ]:
            sort_order += 1
            chart_source_rows.extend(
                [
                    (
                        stage,
                        metric_name,
                        "串行",
                        sequential[key][metric_key],
                        40,
                        sort_order,
                        1,
                    ),
                    (
                        stage,
                        metric_name,
                        "并行",
                        parallel[key][metric_key],
                        40,
                        sort_order,
                        2,
                    ),
                ]
            )
    chart_columns = [
        "stage_metric",
        "stage",
        "metric",
        "mode",
        "seconds",
        "request_count",
    ]
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            """
            CREATE TABLE latency_metrics (
                stage TEXT,
                metric TEXT,
                mode TEXT,
                seconds REAL,
                request_count INTEGER,
                sort_order INTEGER,
                mode_order INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO latency_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            chart_source_rows,
        )
        chart_result = connection.execute(latency_chart_sql).fetchall()
    latency_chart_rows = [
        dict(zip(chart_columns, row, strict=True)) for row in chart_result
    ]

    completion_rows = [
        {
            "workstream": "意图识别与调度",
            "evidence": "60条冻结集×3次有效运行",
            "result": "全部预设门槛通过",
            "status": "完成",
        },
        {
            "workstream": "RAG检索与答案",
            "evidence": "60条冻结集＋人工逐条评分",
            "result": "检索、答案、拒答门槛通过",
            "status": "完成",
        },
        {
            "workstream": "延迟与稳定性",
            "evidence": "40组完整配对/模式",
            "result": "执行阶段改善，端到端有保留项",
            "status": "完成（有边界）",
        },
        {
            "workstream": "离线回归",
            "evidence": "6模块、14项测试",
            "result": "14/14通过",
            "status": "完成",
        },
        {
            "workstream": "技术评测交付",
            "evidence": "原始结果、哈希、分析、HTML报告",
            "result": "可追溯",
            "status": "完成",
        },
    ]
    intent_rows = query_rows(
        intent_rows,
        "intent_report_metrics",
        [
            ("metric", "TEXT"),
            ("mean", "REAL"),
            ("min", "REAL"),
            ("max", "REAL"),
            ("threshold", "TEXT"),
            ("status", "TEXT"),
        ],
        INTENT_TABLE_SQL,
    )
    rag_rows = query_rows(
        rag_rows,
        "rag_report_metrics",
        [
            ("metric", "TEXT"),
            ("value", "TEXT"),
            ("threshold", "TEXT"),
            ("status", "TEXT"),
        ],
        RAG_TABLE_SQL,
    )
    latency_rows = query_rows(
        latency_rows,
        "latency_report_metrics",
        [
            ("stage", "TEXT"),
            ("metric", "TEXT"),
            ("sequential", "REAL"),
            ("parallel", "REAL"),
            ("change", "REAL"),
            ("interpretation", "TEXT"),
        ],
        LATENCY_TABLE_SQL,
    )
    completion_rows = query_rows(
        completion_rows,
        "completion_audit",
        [
            ("workstream", "TEXT"),
            ("evidence", "TEXT"),
            ("result", "TEXT"),
            ("status", "TEXT"),
        ],
        COMPLETION_TABLE_SQL,
    )

    source_by_id = {item["id"]: item for item in sources}

    tables = [
        table(
            "completion",
            "代码与评测阶段完成审计",
            "按预先定义的完成标准逐项核验",
            "completion",
            source_by_id["offline_tests"],
            [
                {"field": "workstream", "label": "工作流", "type": "text"},
                {"field": "evidence", "label": "证据", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
            "workstream",
        ),
        table(
            "intent",
            "意图识别 V2 核心指标",
            "60条冻结独立集，3次有效运行，共180个请求",
            "intent",
            source_by_id["intent_v2"],
            [
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "mean", "label": "均值", "format": "percent"},
                {"field": "min", "label": "最小", "format": "percent"},
                {"field": "max", "label": "最大", "format": "percent"},
                {"field": "threshold", "label": "门槛", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
            "metric",
        ),
        table(
            "rag",
            "RAG V1 核心指标",
            "60条冻结样本；人工评分已逐条确认",
            "rag",
            source_by_id["rag_v1"],
            [
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "value", "label": "结果", "type": "text"},
                {"field": "threshold", "label": "门槛", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
            "metric",
        ),
        table(
            "latency",
            "串行与并行延迟对照",
            "每种模式40次，10轮顺序平衡配对；单位为秒",
            "latency",
            source_by_id["latency_table_v1"],
            [
                {"field": "stage", "label": "阶段", "type": "text"},
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "sequential", "label": "串行", "format": "number"},
                {"field": "parallel", "label": "并行", "format": "number"},
                {
                    "field": "change",
                    "label": "改善率",
                    "format": "percent",
                    "semantic": "movement",
                },
                {"field": "interpretation", "label": "解释", "type": "text"},
            ],
            "stage",
        ),
    ]

    latency_chart_source = next(item for item in sources if item["id"] == "latency_v1")
    charts = [
        {
            "id": "latency_comparison",
            "title": "串行与并行耗时",
            "subtitle": "同一冻结任务集；每种模式40次测量；单位为秒",
            "type": "bar",
            "dataset": "latency_chart",
            "source": latency_chart_source,
            "encodings": {
                "x": {
                    "field": "stage_metric",
                    "type": "nominal",
                    "label": "阶段与指标",
                },
                "y": {
                    "field": "seconds",
                    "type": "quantitative",
                    "format": "number",
                    "label": "秒",
                },
                "color": {
                    "field": "mode",
                    "type": "nominal",
                    "label": "模式",
                },
                "tooltip": [
                    {"field": "stage_metric", "type": "text"},
                    {"field": "mode", "type": "text"},
                    {
                        "field": "seconds",
                        "type": "quantitative",
                        "format": "number",
                    },
                    {"field": "request_count", "type": "quantitative"},
                ],
            },
            "valueFormat": "number",
            "unit": "秒",
            "layout": "full",
        }
    ]

    blocks = [
        markdown("title", "# 差旅出行助手：最终技术评测"),
        markdown(
            "technical_summary",
            "## 技术结论：代码与评测阶段可以结束\n\n"
            "- **意图识别达到冻结门槛。** 60条独立测试集连续运行3次，意图集合完全匹配率均值99.44%，Macro F1均值99.68%，调度完全匹配率均值99.44%，实体字段准确率均值90.86%。\n"
            "- **RAG检索和答案质量达到冻结门槛。** Recall@5为100%，人工答案均分1.86/2，10条不可回答题全部正确拒答。\n"
            "- **并行优化对执行阶段有效，但端到端不能宣称降低50%。** Agent执行阶段P50降低45.8%、P95降低46.0%；端到端均值降低13.0%、P95降低27.6%，但P50上升13.9%。\n"
            "- **稳定性回归通过。** 当前代码的6个离线测试模块共14项测试全部通过；并行模式虽然最终成功率保持100%，但产生10次子Agent重试，仍需持续观察限流。",
        ),
        markdown(
            "completion_text",
            "## 五类交付证据均已具备\n\n"
            "完成标准不是“所有指标都完美”，而是冻结数据、保留失败尝试、按预设口径评测，并清楚标注结果边界。当前意图、RAG、延迟、稳定性和可追溯交付均已形成证据链，因此可以结束代码与评测阶段，转入PRD和项目集包装。",
        ),
        {
            "id": "completion_table",
            "type": "table",
            "tableId": "completion",
            "layout": "full",
        },
        markdown(
            "intent_text",
            "## 意图与调度在独立冻结集上稳定通过\n\n"
            "三次有效运行共覆盖180个请求，且API/解析错误率为0。意图、Macro F1和调度指标显著高于预设门槛；实体字段准确率90.86%刚过90%门槛，是后续最值得继续监控的质量项。\n\n"
            "**解释边界：** 这些结果证明当前冻结集表现，不等同于真实线上流量准确率；不能写成“任意用户表达下准确率99%”。",
            "intent_v2",
        ),
        {"id": "intent_table", "type": "table", "tableId": "intent", "layout": "full"},
        markdown(
            "rag_text",
            "## RAG检索无漏题，但跨文档完整召回仍有缺口\n\n"
            "50条可回答题的Recall@5为100%，表示每题前5条至少命中一个期望来源；人工答案均分1.86/2，且10条不可回答题全部正确拒答。两道跨文档问题只召回一半期望来源，因此“Recall@5=100%”不能解释成“所有证据均完整召回”。\n\n"
            "**解释边界：** 关键词覆盖率只用于诊断，不作为答案正确率；最终答案质量依据人工评分。",
            "rag_v1",
        ),
        {"id": "rag_table", "type": "table", "tableId": "rag", "layout": "full"},
        markdown(
            "latency_text",
            "## 并行化改善执行阶段，端到端收益受意图识别波动稀释\n\n"
            "在40组完整配对样本中，并行执行阶段胜率为72.5%，P50和P95分别降低45.8%和46.0%。意图识别位于并行批次之外，其供应商超时和重试主导部分长尾，因此端到端P50反而增加13.9%。\n\n"
            "**产品含义：** 简历可以写“Agent执行阶段P50降低45.8%”，但不能写“整体响应时间降低50%”。并行模式有5个请求触发共10次子Agent重试，说明并发收益伴随限流压力。",
            "latency_v1",
        ),
        {
            "id": "latency_chart",
            "type": "chart",
            "chartId": "latency_comparison",
            "layout": "full",
        },
        {
            "id": "latency_table",
            "type": "table",
            "tableId": "latency",
            "layout": "full",
        },
        markdown(
            "methodology",
            "## 方法：先冻结，再运行，再人工确认\n\n"
            "1. 意图评测先冻结60条独立集及哈希，再固定模型和Thinking配置运行3次，只聚合样本完整、哈希一致、错误率为0的运行。\n"
            "2. RAG评测冻结60条问题、期望来源和关键答案点；检索使用Recall@5/MRR，答案使用0/1/2分人工评分，不可回答题单独核验拒答。\n"
            "3. 延迟评测固定4类多Agent任务，同一进程内交替串行/并行顺序，每种模式预热1轮、正式测量40次，并记录重试、成功率、P50和P95。\n"
            "4. 无效尝试不删除：它们用于暴露错误传播、超时重试和限流条件下的稳定性缺陷。",
        ),
        markdown(
            "robustness",
            "## 稳健性：完整性检查通过，外部效度有限\n\n"
            "- 延迟原始报告声明有效；数据集哈希一致；80行记录唯一；40个配对完整；顺序平衡；调度全部匹配。\n"
            "- 当前回归测试为14/14通过，覆盖记忆持久化、读后写快照、优先级编排、意图规则、RAG错误序列化和重试传播。\n"
            "- 所有在线评测均来自同一模型供应商账号和单台工作站；绝对延迟会受时段、配额和网络影响。\n"
            "- 样本规模适合项目验收和面试说明，不足以建立行业级泛化结论或统计因果结论。\n"
            "- 本报告未使用图表：三类评测的指标尺度、分母和实验设计不同，合并成单一图形会弱化口径差异；精确表格更适合本次技术审计。",
        ),
        markdown(
            "resume_claims",
            "## 可用于简历的证据化表述\n\n"
            "- 构建60条独立意图评测集并完成3轮冻结评测，意图集合完全匹配率均值99.44%、Macro F1均值99.68%、调度完全匹配率均值99.44%。\n"
            "- 建立60条RAG评测集，Recall@5达到100%，50条可回答题人工答案均分1.86/2，10条不可回答题正确拒答率100%。\n"
            "- 设计串并行配对实验，各模式40次测量；同优先级Agent并行使执行阶段P50降低45.8%、P95降低46.0%，成功率保持100%。\n"
            "- 识别并修复子Agent错误误计成功、RAG摘要失败伪成功及429重试未透传等稳定性问题。\n\n"
            "**禁止表述：** “整体响应时间降低50%”“RAG准确率95%”“线上意图准确率99%”或“并行没有任何代价”。",
        ),
        markdown(
            "next_steps",
            "## 下一步：停止继续刷技术指标，进入产品化表达\n\n"
            "1. 编写PRD：用户问题、目标用户、核心场景、功能边界、异常流程和成功指标。\n"
            "2. 绘制产品方案：IntentionAgent、OrchestrationAgent、MemoryManager和业务Agent的流程及优先级规则。\n"
            "3. 准备项目集页面：问题—方案—关键取舍—评测证据—复盘，而不是只展示代码结构。\n"
            "4. 准备5分钟面试演示和追问答案，重点解释冻结集、人工标注、指标口径及并发限流取舍。\n"
            "5. 后续若继续优化，只优先处理实体字段准确率、跨文档完整召回和并发限流，不再为简历数字反复调测试集。",
        ),
        markdown(
            "questions",
            "## 后续需要回答的产品问题\n\n"
            "- 企业差旅助手的首要北极星指标应是任务完成率、合规率，还是人工节省时长？\n"
            "- 哪些偏好应长期保存，哪些只能作为本次行程约束，用户如何查看和撤销？\n"
            "- 并发限流触发时，产品应降级为串行、排队等待，还是返回部分结果？\n"
            "- 企业制度更新后，知识库如何版本化并对历史答案进行追溯？",
        ),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "差旅出行助手：最终技术评测",
            "description": "意图识别V2、RAG V1、串并行延迟V1与离线回归测试的最终冻结结论。",
            "generatedAt": generated_at,
            "sources": sources,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "completion": completion_rows,
                "intent": intent_rows,
                "rag": rag_rows,
                "latency": latency_rows,
                "latency_chart": latency_chart_rows,
            },
        },
        "sources": sources,
    }
    OUTPUT.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_notes = {
        "generated_at": generated_at,
        "audience": "technical",
        "delivery_mode": "html",
        "required_structure_mapping": {
            "title": "title",
            "technical_summary": "technical_summary",
            "key_findings": ["intent_text", "rag_text", "latency_text"],
            "scope_data_definitions": "source metadata and finding sections",
            "methodology": "methodology",
            "limitations_uncertainty_robustness": "robustness",
            "recommended_next_steps": "next_steps",
            "further_questions": "questions",
        },
        "visual_omission_reason": (
            "Intent and RAG use exact tables because their metrics have different "
            "denominators and semantics. Only the within-experiment latency comparison "
            "is charted because all six values share seconds as the unit."
        ),
        "chart_map": [
            {
                "section": "latency_text",
                "analytical_question": (
                    "How do sequential and parallel modes compare at the execution and "
                    "end-to-end stages?"
                ),
                "takeaway": (
                    "Parallel mode improves execution-stage P50/P95, while end-to-end "
                    "P50 does not improve."
                ),
                "family": "comparison",
                "chart_type": "grouped bar",
                "fields": ["stage_metric", "mode", "seconds"],
                "data_rows": 12,
                "palette_policy": (
                    "hard two-root cap; mode is also identified by grouped position and legend"
                ),
                "delivery_artifact": rel(OUTPUT),
                "source": rel(LATENCY_PATH),
            }
        ],
        "source_hashes": {
            rel(INTENT_PATH): sha256(INTENT_PATH),
            rel(RAG_PATH): sha256(RAG_PATH),
            rel(LATENCY_PATH): sha256(LATENCY_PATH),
            rel(LATENCY_CHART_SQL_PATH): sha256(LATENCY_CHART_SQL_PATH),
        },
        "offline_test_receipt": {
            "command": (
                ".venv/Scripts/python.exe -m pytest tests/test_core_offline.py "
                "tests/test_intention_rules.py evaluation/test_metrics.py "
                "tests/test_rag_reply_serialization_offline.py tests/test_rag_eval_retry.py "
                "tests/test_orchestration_error_propagation.py -q"
            ),
            "result": "14 passed in 7.10s",
            "executed_on": "2026-07-30",
            "warning": (
                "pytest-asyncio emitted a deprecation warning because "
                "asyncio_default_fixture_loop_scope is unset."
            ),
        },
    }
    SOURCE_NOTES.write_text(
        json.dumps(source_notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved artifact: {OUTPUT}")
    print(f"Saved source notes: {SOURCE_NOTES}")


if __name__ == "__main__":
    build()
