# 差旅出行助手｜项目交付包 V1

这是一个基于智谱 GLM 和 AgentScope 的多智能体差旅助手，包含可运行代码、Skill 插件、两层记忆、企业制度 RAG、冻结测试集、正式评测结果和最终技术报告。

当前状态：**代码与技术评测阶段已完成并冻结**。后续工作应转向 PRD、产品方案、项目集页面和面试演示，不再为简历数字反复修改冻结测试集。

## 1. 已实现能力

- IntentionAgent：识别多意图、实体和 Agent 调度优先级。
- OrchestrationAgent：按 Priority 分批执行，同优先级 Agent 并行。
- MemoryManager：管理短期对话、长期记忆和同批次读后写快照。
- Preference Agent：新增、追加和覆盖长期差旅偏好。
- Memory Query Agent：查询历史偏好、对话和差旅记录。
- RAG Knowledge Agent：查询企业差旅制度并返回来源。
- Information Query Agent：联网查询公开旅行信息。
- Event Collection Agent：收集出发地、目的地、日期和时长。
- Itinerary Planning Agent：在 Priority 1 结果基础上生成行程。
- 稳定性机制：超时与 429 退避重试、熔断、错误透传、健康检查。

核心链路：

```text
用户输入
  → IntentionAgent
  → Agent Schedule
  → OrchestrationAgent
  → Priority 1：memory / preference / RAG / information / event
  → Priority 2：itinerary planning
  → 结果聚合与记忆提交
```

## 2. 目录结构

```text
差旅出行助手_项目交付包_V1/
├─ cli.py                         # CLI 入口
├─ config.py                      # 环境变量与运行配置
├─ agents/                        # 意图识别、编排、懒加载
├─ context/                       # 长短期记忆与 MemoryManager
├─ utils/                         # JSON、重试、熔断、Skill 加载
├─ .claude/skills/                # 六类业务 Agent、知识文档和 RAG 缓存
├─ data/models/                   # 本地 BGE 中文嵌入模型
├─ data/memory/                   # 空的本地记忆目录，运行后自动写入
├─ tests/                         # 单元、离线和集成测试
├─ evaluation/
│  ├─ datasets/                   # 三套冻结正式评测集及清单
│  ├─ reports/                    # 正式结果、人工评分、原始证据和 HTML 报告
│  └─ *.py                        # 评测、审计、冻结和分析脚本
├─ docs/项目运行与验收.md
├─ tools/verify_delivery_package.py
├─ requirements*.txt
└─ .env.example
```

未打包内容：

- 原项目的 `.venv`、`__pycache__`、`.pytest_cache`。
- 真实 `.env` 和 API Key。
- 个人会话及历史记忆文件。
- PID、临时进度、调试日志和早期烟雾测试报告。

## 3. 安装

推荐 Python 3.11 或 3.12。在 PowerShell 中进入本目录后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

然后编辑 `.env`，只替换：

```text
ZHIPUAI_API_KEY=你的智谱APIKey
```

如果只需要运行非 RAG 核心能力，可安装体积更小的依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

完整 RAG 依赖位于 `requirements-rag.txt`。本交付包已包含本地嵌入模型和已构建的知识库缓存，不需要重新下载模型文件。

## 4. 运行

先检查模型服务：

```powershell
.\.venv\Scripts\python.exe cli.py health
```

启动交互式 CLI：

```powershell
.\.venv\Scripts\python.exe cli.py
```

可演示的问题：

```text
我之前去过哪些城市？
我以后优先住亚朵，帮我记住。
查询公司一线城市住宿标准，同时记住我喜欢有健身房的酒店。
我明天从成都去北京出差三天，帮我规划行程。
```

## 5. 测试

不调用外部 API 的核心回归测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_core_offline.py `
  tests\test_intention_rules.py `
  evaluation\test_metrics.py `
  tests\test_rag_reply_serialization_offline.py `
  tests\test_rag_eval_retry.py `
  tests\test_orchestration_error_propagation.py -q
```

冻结时结果：**14/14 通过**。

检查交付包是否缺文件、哈希是否变化或包含疑似密钥：

```powershell
.\.venv\Scripts\python.exe tools\verify_delivery_package.py
```

其他 `tests/test_*.py` 包含在线或集成测试，可能调用智谱、搜索服务或本地 RAG，运行前请确认 `.env`、网络和依赖配置。

## 6. 正式评测结果

|模块|正式结果|评测范围|
|---|---:|---|
|意图集合完全匹配率|99.44%|60条冻结集，3轮有效运行|
|Macro F1|99.68%|六类业务意图|
|调度完全匹配率|99.44%|60条冻结集，3轮有效运行|
|实体字段准确率|90.86%|冻结实体接受规则|
|RAG Recall@5|100%|50条可回答题|
|RAG人工答案均分|1.86/2|50条可回答题|
|不可回答题正确拒答率|100%|10条不可回答题|
|并行执行阶段 P50|降低45.8%|串并行各40次，完整配对|
|并行执行阶段 P95|降低46.0%|串并行各40次，完整配对|
|端到端平均耗时|降低13.0%|串并行各40次，完整配对|
|端到端 P50|上升13.9%|并行在该指标更慢|

最终报告：

- `evaluation/reports/final_technical_evaluation.html`
- `evaluation/reports/technical_evaluation_v1.manifest.json`

## 7. 指标表述边界

可以写：

- “构建60条独立意图评测集并完成3轮冻结评测，意图集合完全匹配率均值99.44%、Macro F1均值99.68%。”
- “建立60条RAG评测集，Recall@5达到100%，人工答案均分1.86/2，10条不可回答题正确拒答率100%。”
- “设计串并行配对实验，各模式40次测量；并行编排使 Agent 执行阶段 P50 降低45.8%、P95降低46.0%。”

不能写：

- “整体响应时间降低50%。”
- “RAG准确率95%。”
- “线上所有用户表达的意图识别准确率为99%。”
- “并行执行没有任何代价。”

并行模式在正式实验中产生10次子 Agent 重试，说明并发优化会增加限流压力。

## 8. 安全说明

- `.env.example` 只包含占位符。
- 不要分享或提交 `.env`。
- 用户记忆目录在交付包中为空，避免携带个人对话。
- 如果某个 API Key 曾出现在聊天、README 或版本历史中，应在智谱平台立即作废并重新生成。

详细运行、复现和证据索引见 [项目运行与验收](docs/项目运行与验收.md)。
