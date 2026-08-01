"""
Configuration for the Aligo Multi-Agent System
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    # Environment variables still work when python-dotenv is unavailable.
    pass

# LLM Configuration
# 密钥只从环境变量读取，避免被提交到代码仓库。
LLM_CONFIG = {
    "provider": "zhipu",
    "api_key": os.getenv("ZHIPUAI_API_KEY", "").strip(),
    "model_name": os.getenv("ZHIPUAI_MODEL", "glm-4.7").strip(),
    "base_url": os.getenv(
        "ZHIPUAI_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4",
    ).rstrip("/"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192")),
    # GLM-4.7 默认强制深度思考。普通差旅流程默认关闭以降低延迟。
    "thinking_type": os.getenv(
        "ZHIPUAI_THINKING",
        "disabled",
    ).strip().lower(),
}


def validate_llm_config() -> None:
    """启动前校验模型配置，并给出不泄露密钥的错误信息。"""
    invalid_keys = {
        "",
        "API_KEY",
        "your-api-key-here",
        "replace-with-a-new-key",
    }
    if LLM_CONFIG["api_key"] in invalid_keys:
        raise RuntimeError(
            "未检测到智谱 API Key。请先设置环境变量 ZHIPUAI_API_KEY，"
            "再重新启动程序。"
        )
    if not LLM_CONFIG["model_name"]:
        raise RuntimeError("ZHIPUAI_MODEL 不能为空。")
    if not LLM_CONFIG["base_url"].startswith("https://"):
        raise RuntimeError("ZHIPUAI_BASE_URL 必须是 HTTPS 地址。")
    if LLM_CONFIG["thinking_type"] not in {"enabled", "disabled"}:
        raise RuntimeError(
            "ZHIPUAI_THINKING 只能设置为 enabled 或 disabled。",
        )

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
    # 开启后会额外调用一次 LLM 总结其他会话；默认关闭以缩短首轮响应。
    "enable_long_term_summary": False,
}

# RAG 知识库：嵌入模型（本地路径，无需连 HuggingFace）
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}
