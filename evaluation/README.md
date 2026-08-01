# Aligo 评测工具

本目录用于生成可复现的意图识别、RAG 和响应时间指标。`*.sample.jsonl`
是用于验证评测链路的种子数据，不能直接作为简历中的正式评测结果。

## 1. 离线检查

```powershell
.\.venv\Scripts\python.exe evaluation\test_metrics.py
.\.venv\Scripts\python.exe evaluation\validate_datasets.py
.\.venv\Scripts\python.exe evaluation\run_rag_eval.py
```

第二条命令默认只测试本地检索，不调用智谱 API。

## 2. 意图识别

快速冒烟测试：

```powershell
.\.venv\Scripts\python.exe evaluation\run_intent_eval.py --limit 3
```

运行全部种子数据：

```powershell
.\.venv\Scripts\python.exe evaluation\run_intent_eval.py
```

分析逐例错误与 P50/P95 延迟：

```powershell
.\.venv\Scripts\python.exe evaluation\analyze_intent_results.py
```

只在已保存的模型输出上回放新的确定性后处理规则（不调用 API）：

```powershell
.\.venv\Scripts\python.exe evaluation\replay_intent_postprocessing.py
```

回放结果只能证明后处理逻辑是否修复旧错误，不能替代重新调用模型的正式评测。

代码或 Prompt 修改后，保留旧基线并写入新文件：

```powershell
.\.venv\Scripts\python.exe evaluation\run_intent_eval.py `
  --output evaluation/reports/intent_results_v2.json
```

对比 V1 和 V2：

```powershell
.\.venv\Scripts\python.exe evaluation\compare_intent_runs.py `
  --baseline evaluation/reports/intent_results.json `
  --candidate evaluation/reports/intent_results_v2.json `
  --output evaluation/reports/intent_comparison_v1_v2.json
```

新生成的意图评测结果会记录数据集 SHA-256，用于确认后续实验是否真的使用
同一版冻结数据。旧报告没有该字段，因此实体标注被修改后不能做严格归因。

实体字段口径：

- `origin`：行程或路线的出发地。
- `destination`：行程目的地；对于纯信息查询，也临时承载“查询地点”，不代表
  用户已经形成去该地的行程。
- `duration`：按行程覆盖的日历天数表达，例如周五至周日为3天2晚。
- `other`：完整的未结构化业务语义槽，覆盖偏好值及其极性/增删动作、
  记忆查询对象、企业制度主题、公开信息主题、出差目的和单次行程约束。
  多意图样本必须覆盖每个子意图的关键内容。
- `contains_all` 表示模型输出必须包含全部关键内容；`contains_any` 表示多个
  等价表达命中任意一个即可。
- `contains_all_groups` 表示每个语义组都必须命中，但组内允许等价表达。
  否定极性、删除动作、预算上限等不能只用裸实体词验证。

编排一致性规则：同一 Priority 的智能体读取同一个批次开始前记忆快照，
`OrchestrationAgent` 在所有批次执行完成后统一写入偏好和行程。因此
`memory_query + preference` 可以保持 Priority 1 并行，查询结果仍对应修改前值。

正式评测前，将数据扩展到至少90条，并由人工逐条确认
`expected_intents`、`expected_priorities` 和 `expected_entities`。

### 独立测试集冻结流程

当前候选集包含60条新样本：

```text
evaluation/datasets/intent_eval.holdout.candidate.jsonl
```

先运行自动审计：

```powershell
.\.venv\Scripts\python.exe evaluation\audit_intent_dataset.py
```

然后打开 `evaluation/reports/intent_holdout_review.md`，逐条人工确认。首次模型
评测前可以修正候选集；首次评测后禁止根据模型输出修改答案。

人工复核完成后冻结：

```powershell
.\.venv\Scripts\python.exe evaluation\freeze_intent_dataset.py `
  --reviewer "复核人姓名" `
  --approval I_REVIEWED_ALL_60
```

冻结脚本会生成 `intent_eval.holdout.v1.jsonl` 及 SHA-256 清单，并拒绝覆盖
已经冻结的版本。

### 实体抽取开发回归

冻结集错误归因后，使用独立开发回归集优化实体字段边界：

```text
evaluation/datasets/intent_eval.entity_regression.dev.jsonl
```

开发集可以修正标注和调试规则，但不能作为简历中的独立测试成绩。对旧输出
按当前开发答案和确定性规则重放：

```powershell
.\.venv\Scripts\python.exe evaluation\replay_intent_postprocessing.py `
  --input evaluation/reports/intent_entity_regression_dev_v2_results.json `
  --dataset evaluation/datasets/intent_eval.entity_regression.dev.jsonl `
  --output evaluation/reports/intent_entity_regression_dev_v2_rescored.json
```

优化完成后必须新建、人工复核并冻结盲测 V2，不能重新使用已经查看过预测结果的
V1 证明优化后的泛化能力。

## 3. RAG

正式候选集包含60条问题（50条可回答、10条不可回答，其中5条需要跨文档
取证）。首次模型评测前先执行自动审计：

```powershell
.\.venv\Scripts\python.exe evaluation\audit_rag_dataset.py
```

然后打开 `evaluation/reports/rag_formal_review.md`，逐条核对问题、期望来源、
答案要点和不可回答边界。人工确认后冻结：

```powershell
.\.venv\Scripts\python.exe evaluation\freeze_rag_dataset.py `
  --reviewer "复核人姓名" `
  --approval I_REVIEWED_ALL_60_RAG
```

冻结后先评估 Recall@5、全来源召回率和 MRR，不调用智谱 API：

```powershell
.\.venv\Scripts\python.exe evaluation\run_rag_eval.py `
  --dataset evaluation/datasets/rag_eval.formal.v1.jsonl `
  --top-k 5 `
  --output evaluation/reports/rag_formal_v1_retrieval.json
```

再调用 GLM 生成答案：

```powershell
.\.venv\Scripts\python.exe evaluation\run_rag_eval.py `
  --dataset evaluation/datasets/rag_eval.formal.v1.jsonl `
  --top-k 5 `
  --with-answers `
  --output evaluation/reports/rag_formal_v1_answers.json
```

报告会记录冻结数据集 SHA-256、任一来源命中率、跨文档全来源召回率、
平均来源召回率、MRR、关键词覆盖、拒答率、错误率和429重试次数。

答案生成完成后创建独立的人工评分模板，避免改写原始模型结果：

```powershell
.\.venv\Scripts\python.exe evaluation\review_rag_answers.py `
  --create-template
```

打开 `evaluation/reports/rag_formal_v1_manual_review.jsonl`，根据统一标准填写：

```json
"manual_answer_score_0_to_2": 2
```

- 2：关键点完整且无事实错误
- 1：部分正确或存在明显遗漏
- 0：错误、编造或答非所问

可回答题填写 `answer_score_0_to_2`；不可回答题保持分数为空，并将
`unanswerable_outcome` 填为 `correct_refusal`、`hallucination` 或
`ambiguous`。全部复核后生成最终汇总：

```powershell
.\.venv\Scripts\python.exe evaluation\review_rag_answers.py
```

关键词子串覆盖只是诊断指标，不能代替人工语义评分。简历使用的答案质量数字
必须来自人工评分，并保留逐题评分证据。

## 4. 延迟基准

先跑串行基线：

```powershell
.\.venv\Scripts\python.exe evaluation\run_latency_benchmark.py `
  --mode sequential `
  --label sequential `
  --repeats 4 `
  --warmup 2 `
  --output evaluation/reports/latency_sequential.json
```

再跑并行候选方案：

```powershell
.\.venv\Scripts\python.exe evaluation\run_latency_benchmark.py `
  --mode parallel `
  --label parallel `
  --repeats 4 `
  --warmup 2 `
  --output evaluation/reports/latency_parallel.json
```

对比两个结果：

```powershell
.\.venv\Scripts\python.exe evaluation\compare_latency.py `
  --baseline evaluation/reports/latency_sequential.json `
  --candidate evaluation/reports/latency_parallel.json
```

两次实验必须使用相同模型、Thinking配置、数据集和网络环境。8条种子问题重复4次，
可得到32条测量记录。

## 5. 汇总报告

```powershell
.\.venv\Scripts\python.exe evaluation\generate_report.py
```

报告会检查样本量、人工评分和实验可比性。只有所有检查通过后，相关数字才适合写入简历。
