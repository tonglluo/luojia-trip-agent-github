# 意图识别 V2 三轮正式评测报告

- 冻结集：`evaluation/datasets/intent_eval.holdout.v2.jsonl`
- SHA-256：`3c89c80ab5fbd92d28269571b3f4e7adb8699f8c3164561f86c29350cd0ba526`
- 模型 / Thinking：glm-4.7 / disabled
- 有效运行：3 次，每次 60 条
- 总体门槛：通过

## 核心指标

|指标|均值|最小|最大|门槛|结论|
|---|---:|---:|---:|---:|---|
|意图集合完全匹配率|99.44%|98.33%|100.00%|≥90%|通过|
|Macro F1|99.68%|99.05%|100.00%|≥95%|通过|
|调度完全匹配率|99.44%|98.33%|100.00%|≥95%|通过|
|实体字段准确率|90.86%|89.52%|92.74%|≥90%|通过|
|错误率|0.00%|0.00%|0.00%|≤0%|通过|

## 逐类 F1

|意图|F1均值|最小|最大|
|---|---:|---:|---:|
|event_collection|99.05%|97.14%|100.00%|
|information_query|100.00%|100.00%|100.00%|
|itinerary_planning|99.05%|97.14%|100.00%|
|memory_query|100.00%|100.00%|100.00%|
|preference|100.00%|100.00%|100.00%|
|rag_knowledge|100.00%|100.00%|100.00%|

## 错误归因

- 至少一次失败的样本：16
- 三轮共同失败：8
- 随机波动失败：8

|ID|类别|类型|意图失败轮次|调度失败轮次|实体失败轮次|实体字段|
|---|---|---|---|---|---|---|
|holdout_v2_006|single_itinerary|共同|—|—|[1, 2, 3]|other|
|holdout_v2_010|single_preference|共同|—|—|[1, 2, 3]|other|
|holdout_v2_015|single_preference|共同|—|—|[1, 2, 3]|other|
|holdout_v2_016|single_memory|共同|—|—|[1, 2, 3]|other|
|holdout_v2_023|single_rag|共同|—|—|[1, 2, 3]|other|
|holdout_v2_032|single_information|共同|—|—|[1, 2, 3]|duration|
|holdout_v2_038|multi_preference_itinerary|共同|—|—|[1, 2, 3]|other|
|holdout_v2_048|multi_preference_itinerary|共同|[1]|[1]|[1, 2, 3]|other|
|holdout_v2_011|single_preference|波动|—|—|[2]|other|
|holdout_v2_018|single_memory|波动|—|—|[1]|other|
|holdout_v2_020|single_memory|波动|—|—|[1, 2]|other|
|holdout_v2_025|single_rag|波动|—|—|[3]|other|
|holdout_v2_033|single_information|波动|—|—|[1]|other|
|holdout_v2_035|single_information|波动|—|—|[2]|other|
|holdout_v2_037|multi_preference_itinerary|波动|—|—|[1, 2]|other|
|holdout_v2_047|multi_rag_preference|波动|—|—|[1]|other|

## 运行说明

- 有效请求数：180
- 延迟均值 / P50 / P95：23.41s / 15.99s / 64.48s
- 所有尝试均保留；只有数据集哈希一致、样本完整且错误率为 0 的运行进入聚合。
- 本报告只证明当前冻结测试集上的表现，不等同于线上真实流量准确率。
