"""
测试事项收集智能体
"""
import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from agents.lazy_agent_registry import LazyAgentRegistry
from config import LLM_CONFIG, validate_llm_config


async def test_event_collection_agent():
    """测试事项收集智能体"""

    print("初始化 AgentScope...")
    validate_llm_config()
    init_agentscope()

    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        generate_kwargs={
            "temperature": LLM_CONFIG.get("temperature", 0.7),
            "max_tokens": LLM_CONFIG.get("max_tokens", 2000),
        },
    )
    registry = LazyAgentRegistry(model=model, cache={})
    agent = registry["event_collection"]

    test_cases = [
        "我要从北京去上海出差3天",
        "下周一从杭州出发去深圳，周五回来",
        "去上海玩",  # 信息不完整的情况
        "3月15日从北京到上海，3月18日返回北京，出差",
    ]

    for i, query in enumerate(test_cases, 1):
        print("\n" + "="*70)
        print(f"测试 {i}")
        print("="*70)
        print(f"用户查询: {query}")
        print()

        msg = Msg(name="User", content=query, role="user")
        result = await agent.reply(msg)

        # 显示结果
        content = json.loads(result.content)
        print("【提取结果】")

        if content.get("origin"):
            print(f"  ✓ 出发地: {content['origin']}")
        if content.get("destination"):
            print(f"  ✓ 目的地: {content['destination']}")
        if content.get("start_date"):
            print(f"  ✓ 出发日期: {content['start_date']}")
        if content.get("end_date"):
            print(f"  ✓ 返程日期: {content['end_date']}")
        if content.get("duration_days"):
            print(f"  ✓ 行程天数: {content['duration_days']}天")
        if content.get("return_location"):
            print(f"  ✓ 返程地: {content['return_location']}")
        if content.get("trip_purpose"):
            print(f"  ✓ 行程目的: {content['trip_purpose']}")

        print(f"\n  已提取: {content.get('extracted_count', 0)}/7 项信息")

        if content.get("missing_info"):
            print(f"  ⚠ 缺失信息: {', '.join(content['missing_info'])}")


if __name__ == "__main__":
    print("="*70)
    print("事项收集智能体测试")
    print("="*70)
    asyncio.run(test_event_collection_agent())
    print("\n测试完成！")
