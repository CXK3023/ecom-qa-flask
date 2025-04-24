# app/services/openai_service.py (移除默认模型)
import os
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
from dotenv import load_dotenv, find_dotenv
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("--- 尝试强制加载 .env 文件 (override=True) ---")
load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

logger.info(f"--- DEBUG: 配置客户端使用的 API Key: {api_key[:5]}...{api_key[-4:] if api_key else 'None'} ---")
logger.info(f"--- DEBUG: 配置客户端使用的 API Base: {api_base} ---")

client = None
if api_key and api_base:
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        logger.info(f"--- OpenAI Client 创建成功，使用代理: {api_base} ---")
    except Exception as e:
        logger.error(f"!!! 创建 OpenAI Client 时出错: {e} !!!", exc_info=True)
else:
    if not api_key: logger.error("!!! OpenAI Client 未创建: 缺少 OPENAI_API_KEY !!!")
    if not api_base: logger.error("!!! OpenAI Client 未创建: 缺少 OPENAI_API_BASE (代理地址) !!!")


# --- 定义调用函数 (修改版 - 移除 model 默认值) ---
def get_openai_response(messages: list, model: str): # <--- 修改：移除 model 的默认值
    """
    (多轮对话 + 动态模型版) 调用 OpenAI Chat Completion API 获取回复。

    Args:
        messages (list): 包含对话历史的消息列表。
        model (str): 要使用的 AI 模型名称 (由调用方提供)。

    Returns:
        str: AI 的回复文本，或者在出错时返回错误提示信息。
    """
    if not client:
        logger.error("!!! AI 服务客户端未初始化，无法发送请求 !!!")
        return "抱歉，AI 服务客户端未能正确初始化，请检查后台日志和配置。"

    if not model: # 增加检查，确保调用方提供了模型名称
        logger.error("!!! 调用 get_openai_response 时未提供模型名称 (model 参数) !!!")
        return "抱歉，系统内部错误：未指定要使用的 AI 模型。"

    if messages:
        logger.info(f"--- 向 OpenAI 发送 {len(messages)} 条消息 (模型: {model})。最后一条 ({messages[-1].get('role')}): {str(messages[-1].get('content'))[:80]}... ---")
    else:
        logger.error("!!! 尝试调用 get_openai_response 但传入的 messages 列表为空 !!!")
        return "抱歉，没有收到任何对话内容。"

    try:
        response = client.chat.completions.create(
            model=model, # <--- 使用传入的 model 参数
            messages=messages,
            max_tokens=300
        )

        logger.debug(f"--- DEBUG: 原始 API 响应类型: {type(response)} ---")

        if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message') and response.choices[0].message:
             if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content is not None:
                 answer = response.choices[0].message.content.strip()
                 logger.info(f"--- 收到 OpenAI ({model}) 回复: {answer[:100]}... ---")
                 return answer
             else:
                 logger.error(f"!!! AI ({model}) 响应中 'message' 对象缺少 'content' 属性或为空: {response.choices[0].message} !!!")
                 return "抱歉，AI 返回了有效的响应结构，但内容为空。"
        else:
             error_message_from_response = str(response)
             logger.error(f"!!! AI ({model}) 响应格式不符合预期: {error_message_from_response} !!!")
             detail = error_message_from_response if len(error_message_from_response) < 100 else f"{error_message_from_response[:100]}..."
             return f"抱歉，AI 服务返回了意外的响应格式({type(response).__name__})，可能是代理错误。详情: {detail}"

    except AuthenticationError as e:
        logger.error(f"!!! OpenAI ({model}) 认证错误: {e} !!!", exc_info=True)
        return "抱歉，连接 AI 服务时发生认证错误，请检查 API Key 是否正确或有权限。"
    except APIConnectionError as e:
        logger.error(f"!!! OpenAI ({model}) 连接错误: {e} !!!", exc_info=True)
        return "抱歉，无法连接到 AI 服务，请检查网络连接或代理地址是否配置正确。"
    except RateLimitError as e:
        logger.error(f"!!! OpenAI ({model}) 速率限制错误: {e} !!!", exc_info=True)
        return "抱歉，提问太快了，请稍后再试。"
    except Exception as e:
        logger.error(f"!!! 调用 OpenAI API ({model}) 时发生未知错误: {e} !!!", exc_info=True)
        print(traceback.format_exc())
        return f"抱歉，与 AI 对话时遇到了未知问题 ({type(e).__name__} - 使用模型 {model})，请查看后台日志了解详情。"