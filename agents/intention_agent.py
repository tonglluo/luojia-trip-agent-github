"""
意图识别智能体 IntentionRecognitionAgent
职责：准确识别用户意图，并进行智能体调度

核心功能：
1. 多意图识别和分类：融合上下文对模糊意图进行消歧
2. 智能体调度决策：基于预定义的触发条件和业务规则，根据识别结果决定调用哪些子智能体
3. Query改写：标准化用户口语化的query输入，补全上下文信息，提取和重组关键信息
4. 显示推理：输出的两段式结构（推理过程 + JSON决策），提升意图识别准确度

架构：
- 使用单一LLM（用户配置的模型）
- 输入：用户query（自然语言）
- 输出：推理过程生成（包含reasoning+原因） + 多意图识别（原因） + 智能Query改写 + 构建结构化决策
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List
import json
import logging
import re
from utils.skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class IntentionAgent(AgentBase):
    """意图识别智能体（IntentionRecognitionAgent）"""

    VALID_INTENTS = {
        "itinerary_planning",
        "memory_query",
        "preference",
        "rag_knowledge",
        "information_query",
        "event_collection",
    }
    INTENT_PRIORITIES = {
        "memory_query": 1,
        "event_collection": 1,
        "preference": 1,
        "information_query": 1,
        "rag_knowledge": 1,
        "itinerary_planning": 2,
    }

    def __init__(self, name: str = "IntentionRecognitionAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.conversation_history = []
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        意图识别主流程
        1. 推理过程生成
        2. 多意图识别
        3. 智能Query改写
        4. 构建结构化决策
        """
        if x is None:
            return Msg(name=self.name, content=json.dumps({}), role="assistant")

        # 获取用户查询
        if isinstance(x, list):
            user_query = x[-1].content if x else ""
            # 提取历史对话，保留角色信息
            self.conversation_history = []
            for msg in x[:-1]:
                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                    # 区分处理不同角色的消息
                    if msg.role == "system":
                        # 长期记忆（system）- 完整保留，不截断
                        self.conversation_history.append(f"[系统记忆]\n{msg.content}")
                    else:
                        # 对话历史（user/assistant）- 适当截断但保留更多信息
                        role_name = "用户" if msg.role == "user" else "助手"
                        content = msg.content[:800] if len(msg.content) > 800 else msg.content
                        if len(msg.content) > 800:
                            content += "..."
                        self.conversation_history.append(f"{role_name}: {content}")
        else:
            user_query = x.content

        # 构建上下文
        # 策略：长期记忆始终保留，短期对话全部保留（已在 cli.py 控制数量）
        context_parts = []
        system_memory = None
        dialogue_history = []

        for item in self.conversation_history:
            if item.startswith("[系统记忆]"):
                system_memory = item  # 保存长期记忆
            else:
                dialogue_history.append(item)  # 保存对话历史

        # 组装上下文：长期记忆 + 全部对话
        if system_memory:
            context_parts.append(system_memory)
        if dialogue_history:
            context_parts.extend(dialogue_history) 

        context_str = "\n".join(context_parts) if context_parts else "无历史对话"

        # 获取当前时间
        from datetime import datetime
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 动态获取 Skills 描述
        skill_mapping = {
            "memory-query": "memory_query",
            "plan-trip": "itinerary_planning", 
            "preference": "preference",
            "query-info": "information_query",
            "ask-question": "rag_knowledge",
            "event-collection": "event_collection"
        }
        
        dynamic_skills_prompt = self.skill_loader.get_skill_prompt(skill_mapping)
        
        # 构建意图识别Prompt
        prompt = f"""你是一个高级意图识别专家（IntentionRecognitionAgent）。请分析用户查询，识别意图并输出结构化的决策。
        
【当前时间】
{current_time} {weekday}
（重要：当用户说"2月28日"或"明天"等相对时间时，请根据当前时间进行推断完整日期）

【用户Query】
{user_query}

【对话历史上下文】
{context_str}

【可调度的子智能体 (Skills)】
{dynamic_skills_prompt}

【重要 - 意图区分原则】
请基于语义理解判断意图，不要机械匹配关键词。同一个词在不同语境下可能对应不同意图：
- "我去过北京吗？" → memory_query（询问自己的历史）
- "北京怎么样？" / "北京有什么好玩的？" → information_query（询问客观信息）
- "我想去北京" → itinerary_planning（规划未来行程）

优先级规则：
- memory_query 优先于 information_query（当问题涉及用户自己的历史时）
- 如果用户明确询问"我的"、"我过去的"，必须识别为 memory_query

【任务要求】
请按以下步骤进行分析：

**第1步：推理过程生成**
- 分析用户query的核心诉求
- 识别query中的关键实体和意图信号
- 判断是否需要结合对话历史进行消歧
- 说明如何融合上下文信息进行推理

**第2步：多意图识别（原因）**
- 只识别用户明确表达的业务意图，不要因为某个智能体“可能有帮助”就增加意图
- 为每个意图分配置信度（0-1之间）
- 说明为什么识别出该意图的原因

**第3步：智能Query改写**
- 识别口语化表达，进行标准化
- 补全省略的上下文信息
- 提取和重组关键信息

**第4步：构建结构化决策**
- 基于识别的意图，决定调用哪些子智能体
- 说明调用顺序和优先级
- 输出结构化的调用策略

【输出格式要求】
必须严格按照以下JSON格式输出（**只输出JSON，不要有其他文本**）：

{{
    "reasoning": "这里是详细的推理过程，包含第1步的分析，说明如何理解用户query，如何结合上下文，如何识别意图信号",

    "intents": [
        {{
            "type": "意图类型（如：itinerary_planning, preference_collection, information_query等）",
            "confidence": 0.95,
            "description": "该意图的具体说明",
            "reason": "为什么识别出该意图的原因"
        }}
    ],

    "key_entities": {{
        "origin": "出发地（如果有）",
        "destination": "目的地（如果有）",
        "date": "日期（如果有）",
        "duration": "时长（如果有）",
        "other": "完整记录所有未被上述字段承载的意图关键信息，包括偏好值及其肯定/否定/增删动作、记忆查询对象、制度主题、实时信息主题、出差目的和单次行程约束；多意图时不能遗漏任一子意图的关键内容"
    }},

    "rewritten_query": "标准化、补全后的查询内容",

    "agent_schedule": [
        {{
            "agent_name": "子智能体名称",
            "priority": 1,
            "reason": "调用该智能体的原因和依据",
            "expected_output": "期望该智能体提供什么输出"
        }}
    ]
}}

【重要提示 - 优先级设置规则】
优先级数字相同的智能体会**并行执行**，不同优先级按顺序批次执行。

**所有智能体优先级分组：**

**Priority 1（并行执行）- 信息收集类：**
- memory_query: 记忆查询智能体
- event_collection: 事项收集智能体
- preference: 偏好管理智能体
- information_query: 信息查询智能体（联网搜索）
- rag_knowledge: RAG知识库智能体（查询企业知识库）

**Priority 2（依赖 Priority 1）- 行程规划类：**
- itinerary_planning: 行程规划智能体（需要事项收集的结果）

**说明：**
- Priority 1 的智能体都是信息获取，互不依赖，可并行执行提升速度
- Priority 2 的智能体需要使用 Priority 1 收集的信息
- intents.type 和 agent_schedule.agent_name 只能使用上面列出的六个名称，禁止生成 greeting、chat 等新名称
- 意图与调度必须一一对应：intents 中的每个意图都必须出现在 agent_schedule，agent_schedule 中也不能出现 intents 没有的智能体
- 只要识别到 itinerary_planning，就必须同时识别并调度 event_collection；两者优先级固定为 1 和 2
- preference 只在用户明确声明、增加、删除或修改个人偏好时触发。不能为了让行程更个性化而主动查询偏好，也不能仅因行程规划而触发 preference
- “以后优先”“每次都要”“请记下来”属于长期偏好；“这次想要轻松路线”“本次以室内为主”只是单次行程约束，不触发 preference
- 询问“我以前的偏好是什么”属于 memory_query，不是 preference；如果同一句还要求“改成/再加/删除”，才同时触发 preference
- 当上一轮助手正在询问要记录的偏好时，“不要临街”“靠过道”等省略回答仍属于 preference
- information_query 只处理差旅相关的实时或公开信息；rag_knowledge 只处理企业差旅制度/内部知识。闲聊、数学等非差旅请求返回空 intents 和空 agent_schedule
- key_entities 的 origin、destination、duration 只填写规范化实体值，例如“成都”“北京”“三天”；不要附加“来自上下文”“可能为”等解释
- 对 information_query、rag_knowledge、memory_query，用户查询的地点也必须写入 destination；即使用户没有形成行程，也不能只把“地点/城市”写进 other。省略地点时应从对话历史继承
- 当天往返、当日往返、一日往返的 duration 统一写为“一天”
- other 必须逐项保留用户明确说出的查询对象、目的、偏好强度、肯定/否定/增删动作和单次约束；不能因为相关内容可从 intent 或其他词推断而省略
- 同一句中的地点若同时修饰政策查询和后续行程规划，应把它作为该行程的 destination，除非用户明确给出其他目的地
- 示例：用户说"我要从天津去北京，喜欢住汉庭"
  → Priority 1: preference + event_collection（并行）
  → Priority 2: itinerary_planning（使用 Priority 1 的结果）

请开始分析，直接输出JSON：
"""

        # 调用LLM进行意图识别
        try:
            # 构建符合OpenAI格式的messages
            messages = [
                {"role": "system", "content": "你是一个高级意图识别专家。只输出JSON格式的结果，不要输出其他文本。"},
                {"role": "user", "content": prompt}
            ]
            response = await self.model(messages)

            # 获取响应文本 - 处理异步生成器
            text = ""
            if hasattr(response, '__aiter__'):
                # 异步生成器，需要迭代获取内容
                async for chunk in response:
                    if isinstance(chunk, str):
                        text = chunk
                    elif hasattr(chunk, 'content'):
                        if isinstance(chunk.content, str):
                            text = chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text = item.get('text', '')
            elif hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'content'):
                text = response.content
            elif isinstance(response, dict) and 'content' in response:
                text = response['content']
            else:
                text = str(response) if response else ""

            # 清理文本
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

            # 解析JSON
            try:
                result = json.loads(text)
            except json.JSONDecodeError as e1:
                # 如果直接解析失败，尝试提取JSON
                start_idx = text.find('{')
                end_idx = text.rfind('}')

                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx+1]
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON parse failed. Text sample: {json_str[:100]}")
                        raise ValueError(f"Failed to parse JSON. Error: {e2}")
                else:
                    raise ValueError(f"No JSON found in response. Parse error: {e1}")

            result = self._normalize_result(
                result,
                user_query,
                context_str,
            )

        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            # 意图识别是整个调度链路的入口。失败时必须终止，不能默认
            # 路由到 information_query，否则会把行程规划误当成网络搜索。
            raise RuntimeError(f"意图识别失败: {e}") from e

        # 将结果转换为JSON字符串，因为Msg的content必须是字符串
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    @classmethod
    def _has_explicit_preference_request(
        cls,
        user_query: str,
        context: str = "",
    ) -> bool:
        """Return whether the user is explicitly writing a preference."""
        mutation_pattern = (
            r"记住|记下来|记为|保存|保存为|改成|改为|设为|设置为|"
            r"添加|新增|删除|取消|移除|再加"
        )
        asks_about_existing = bool(
            re.search(
                (
                    r"(?:以前|之前|原来|哪些|什么|是否|有没有).{0,20}"
                    r"(?:偏好|喜欢|保存)"
                    r"|(?:偏好|喜欢|保存).{0,20}(?:哪些|什么|吗|是什么)"
                ),
                user_query,
            ),
        )
        if asks_about_existing and not re.search(
            mutation_pattern,
            user_query,
        ):
            return False

        patterns = (
            r"(?:我|本人).{0,12}(?:喜欢|偏爱|习惯|常住|常坐|只住|吃素)",
            mutation_pattern,
            (
                r"(?:以后|今后|每次|一律|都按).{0,30}"
                r"(?:安排|选择|选|预订|订|优先|注意|按|必须)"
            ),
            r"(?:把|将).{0,30}(?:改成|改为|设为|设置为)",
            r"(?:再|另外)?(?:加|添加|新增|删除|移除).{0,20}",
            r"(?:常用|固定|长期|住宿预算|酒店预算).{0,20}",
            r"每(?:晚|夜).{0,12}(?:以内|不超过|不要超过|上限)",
            r"(?:不再偏好|不喜欢).{0,20}",
        )
        if any(re.search(pattern, user_query) for pattern in patterns):
            return True

        context_requests_preference = bool(
            re.search(r"偏好|习惯|希望记录|设置", context),
        )
        elliptical_preference_answer = bool(
            re.search(
                r"不要|避免|优先|靠窗|靠过道|无烟|不临街|安静|远离",
                user_query,
            ),
        )
        return context_requests_preference and elliptical_preference_answer

    @classmethod
    def _has_internal_knowledge_request(cls, user_query: str) -> bool:
        """Return whether the request explicitly targets enterprise knowledge."""
        return bool(
            re.search(
                r"公司|企业|内部|差旅制度|报销|审批|预订规定|应急流程",
                user_query,
            ),
        )

    @classmethod
    def _should_recover_preference_intent(
        cls,
        user_query: str,
        context: str = "",
    ) -> bool:
        """Recover only explicit, high-precision preference writes."""
        patterns = (
            r"(?:酒店|住宿)预算.{0,16}每(?:晚|夜)",
            r"每(?:晚|夜).{0,16}(?:预算|以内|不超过|不要超过|上限)",
            (
                r"(?:删除|取消|移除|修改|更新|改成|改为).{0,20}"
                r"(?:偏好|习惯)"
            ),
            (
                r"(?:偏好|习惯).{0,20}"
                r"(?:删除|取消|移除|修改|更新|改成|改为)"
            ),
            (
                r"(?:以后|今后|每次|一律|都按).{0,30}"
                r"(?:优先|避免|不订|选择|安排|注意)"
            ),
        )
        if any(re.search(pattern, user_query) for pattern in patterns):
            return True
        return bool(
            re.search(r"偏好|习惯|希望记录|设置", context)
            and re.search(
                r"不要|避免|优先|靠窗|靠过道|无烟|不临街|安静|远离",
                user_query,
            )
        )

    @classmethod
    def _is_travel_information_request(
        cls,
        user_query: str,
        result: dict,
    ) -> bool:
        """Return whether a standalone public-information request is in scope."""
        entities = result.get("key_entities", {})
        if isinstance(entities, dict) and (
            entities.get("origin") or entities.get("destination")
        ):
            return True
        return bool(
            re.search(
                (
                    r"天气|下雨|温度|限行|交通|航班|机票|火车|高铁|"
                    r"机场|车站|酒店|住宿|景点|参观|展会|城市|路线|"
                    r"出行|差旅|出差|旅游|打车|签证|附近|餐厅|美食"
                ),
                user_query,
            ),
        )

    @classmethod
    def _normalize_result(
        cls,
        result: dict,
        user_query: str,
        context: str = "",
    ) -> dict:
        """Enforce routing invariants after the probabilistic LLM decision."""
        if not isinstance(result, dict):
            raise ValueError("Intent result must be a JSON object")

        intents_by_type = {}
        for item in result.get("intents", []):
            if not isinstance(item, dict):
                continue
            intent_type = item.get("type")
            if intent_type in cls.VALID_INTENTS:
                intents_by_type.setdefault(intent_type, item)

        scheduled_names = {
            item.get("agent_name")
            for item in result.get("agent_schedule", [])
            if isinstance(item, dict)
        }
        if (
            "itinerary_planning" in scheduled_names
            and "itinerary_planning" not in intents_by_type
        ):
            intents_by_type["itinerary_planning"] = {
                "type": "itinerary_planning",
                "confidence": 0.8,
                "description": "用户请求规划行程",
                "reason": "由行程规划调度结果补全",
            }

        if "itinerary_planning" in intents_by_type:
            intents_by_type.setdefault(
                "event_collection",
                {
                    "type": "event_collection",
                    "confidence": 1.0,
                    "description": "提取行程规划所需的结构化事项",
                    "reason": "行程规划的固定前置步骤",
                },
            )
        else:
            intents_by_type.pop("event_collection", None)

        has_explicit_preference = cls._has_explicit_preference_request(
            user_query,
            context,
        )
        if (
            "preference" in intents_by_type
            and not has_explicit_preference
        ):
            intents_by_type.pop("preference")
        elif (
            "preference" not in intents_by_type
            and cls._should_recover_preference_intent(
                user_query,
                context,
            )
        ):
            intents_by_type["preference"] = {
                "type": "preference",
                "confidence": 1.0,
                "description": "用户明确写入或修改长期偏好",
                "reason": "由高精度偏好写入规则补全",
            }

        if (
            "rag_knowledge" in intents_by_type
            and "information_query" in intents_by_type
            and not cls._has_internal_knowledge_request(user_query)
        ):
            intents_by_type.pop("rag_knowledge")

        if (
            set(intents_by_type) == {"information_query"}
            and not cls._is_travel_information_request(user_query, result)
        ):
            intents_by_type.pop("information_query")

        intent_order = (
            "memory_query",
            "event_collection",
            "preference",
            "information_query",
            "rag_knowledge",
            "itinerary_planning",
        )
        result["intents"] = [
            intents_by_type[name]
            for name in intent_order
            if name in intents_by_type
        ]

        original_schedule = {
            item.get("agent_name"): item
            for item in result.get("agent_schedule", [])
            if isinstance(item, dict)
            and item.get("agent_name") in cls.VALID_INTENTS
        }
        normalized_schedule = []
        for name in intent_order:
            if name not in intents_by_type:
                continue
            item = dict(original_schedule.get(name, {}))
            item["agent_name"] = name
            item["priority"] = cls.INTENT_PRIORITIES[name]
            item.setdefault("reason", "根据已识别的用户意图进行调度")
            item.setdefault("expected_output", "返回该意图对应的处理结果")
            normalized_schedule.append(item)
        result["agent_schedule"] = normalized_schedule
        return result
