"""
LLM流式响应工具模块
简化版本，专注于DeepSeek API集成
"""

import os
import logging
from typing import AsyncGenerator, List, Dict
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

# 初始化客户端
_client = None

def get_client() -> AsyncOpenAI:
    """获取DeepSeek客户端"""
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        _client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30
        )
    return _client

async def stream_llm(messages: List[Dict[str, str]], 
                    model: str = DEFAULT_MODEL,
                    temperature: float = DEFAULT_TEMPERATURE,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    **kwargs) -> AsyncGenerator[str, None]:
    """
    流式LLM响应
    
    Args:
        messages: 消息列表
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        **kwargs: 额外的API参数
        
    Yields:
        str: 流式响应的文本内容
    """
    try:
        client = get_client()
        
        # 创建流式响应
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 流式返回内容
        async for chunk in stream:
            if (chunk.choices and 
                chunk.choices[0].delta and 
                chunk.choices[0].delta.content is not None):
                yield chunk.choices[0].delta.content
                
    except APIConnectionError as e:
        logger.error(f"API连接错误: {e}")
        yield "抱歉，网络连接出现问题，请检查网络连接后重试。"
    except RateLimitError as e:
        logger.error(f"API速率限制: {e}")
        yield "抱歉，请求过于频繁，请稍后再试。"
    except APIError as e:
        logger.error(f"API错误: {e}")
        yield "抱歉，API服务暂时不可用，请稍后再试。"
    except Exception as e:
        logger.error(f"未知错误: {e}")
        yield "抱歉，系统遇到未知错误，请稍后再试。"

async def get_llm_response(messages: List[Dict[str, str]], 
                          **kwargs) -> str:
    """
    获取非流式LLM响应
    
    Args:
        messages: 消息列表
        **kwargs: 额外的API参数
        
    Returns:
        str: 完整的响应文本
    """
    full_response = ""
    async for chunk in stream_llm(messages, **kwargs):
        full_response += chunk
    return full_response

# 测试函数
async def test_stream():
    """测试流式响应"""
    messages = [{"role": "user", "content": "你好！请介绍一下你自己。"}]
    
    print("流式响应测试:")
    async for chunk in stream_llm(messages):
        print(chunk, end="", flush=True)
    print("\n")

async def test_complete():
    """测试完整响应"""
    messages = [{"role": "user", "content": "你好！请用一句话介绍你自己。"}]
    
    print("完整响应测试:")
    response = await get_llm_response(messages)
    print(response)

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 检查API密钥
        if not DEEPSEEK_API_KEY:
            print("警告: 未设置DEEPSEEK_API_KEY环境变量")
            print("请设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
            return
        
        # 运行测试
        await test_stream()
        await test_complete()
    
    asyncio.run(main())