# app/services/openai_service.py (强制覆盖 + 显式 Client + 增强调试)
import os
# 显式导入 OpenAI 类和可能用到的错误类型
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
# 确保导入了 find_dotenv 和 load_dotenv
from dotenv import load_dotenv, find_dotenv
import traceback # 用于打印完整错误栈

# 使用 find_dotenv() 找到 .env 文件路径，并设置 override=True
# 这会强制加载 .env 文件中的值，即使系统环境变量已存在
print("--- 尝试强制加载 .env 文件 (override=True) ---")
load_dotenv(find_dotenv(), override=True)

# --- 读取环境变量 (现在应该优先来自 .env) ---
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE") # 仍然从 .env 读取

# --- 调试打印，确认读取到的值 ---
print(f"--- DEBUG: 配置客户端使用的 API Key: {api_key[:5]}...{api_key[-4:] if api_key else 'None'} ---")
print(f"--- DEBUG: 配置客户端使用的 API Base: {api_base} ---")

# --- 创建 OpenAI 客户端实例 ---
client = None
if api_key and api_base:
    try:
        # 使用从 .env 读取到的 key 和 base_url 创建客户端
        client = OpenAI(api_key=api_key, base_url=api_base)
        print(f"--- OpenAI Client 创建成功，使用代理: {api_base} ---")
    except Exception as e:
        print(f"!!! 创建 OpenAI Client 时出错: {e} !!!")
        # 如果创建客户端就失败，后面调用会出错
else:
    if not api_key:
        print("!!! OpenAI Client 未创建: 缺少 OPENAI_API_KEY !!!")
    if not api_base:
        print("!!! OpenAI Client 未创建: 缺少 OPENAI_API_BASE (代理地址) !!!")

# --- 定义调用函数 ---
def get_openai_response(prompt_message, model="gpt-4.1"): # 模型保持一致
    """
    (重构版 + 增强调试) 调用 OpenAI Chat Completion API 获取回复。

    Args:
        prompt_message (str): 发送给 AI 的用户问题或完整提示。
        model (str): 要使用的 AI 模型名称。

    Returns:
        str: AI 的回复文本，或者在出错时返回错误提示信息。
    """
    # 检查客户端是否成功创建
    if not client:
        return "抱歉，AI 服务客户端未能正确初始化，请检查后台日志和配置。"

    print(f"--- 向 OpenAI 发送消息 (通过显式 Client): {prompt_message[:100]}... ---")

    try:
        # !!! 使用 client 实例来调用 API !!!
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个友好的电商问答助手。请用简洁明了的语言回答问题。"},
                {"role": "user", "content": prompt_message}
            ],
            max_tokens=150
        )

        # --- 打印原始响应，帮助调试 ---
        print(f"--- DEBUG: 原始 API 响应类型: {type(response)} ---")
        print(f"--- DEBUG: 原始 API 响应内容(部分): {str(response)[:500]}... ---") # 打印部分内容防止过长
        # --- 调试打印结束 ---

        # 提取回答 (增加检查，防止 response 是字符串时出错)
        # 使用 hasattr 检查 response 是否真的有 choices 属性
        if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message') and response.choices[0].message:
             # 确保 message 对象也有 content 属性
             if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content is not None:
                 answer = response.choices[0].message.content.strip()
                 print(f"---收到 OpenAI 回复: {answer[:100]}... ---")
                 return answer
             else:
                 # 处理 message 对象存在但 content 为空或不存在的情况
                 print(f"!!! AI 响应中 'message' 对象缺少 'content' 属性或为空: {response.choices[0].message} !!!")
                 return "抱歉，AI 返回了有效的响应结构，但内容为空。"
        else:
             # 如果 response 没有 choices 属性，很可能它就是错误字符串本身或非预期结构
             error_message_from_response = str(response) # 转换为字符串以防万一
             print(f"!!! AI 响应格式不符合预期，可能是错误消息或非预期结构: {error_message_from_response} !!!")
             # 返回一个更具体的错误信息给前端，包含部分原始响应
             detail = error_message_from_response if len(error_message_from_response) < 100 else f"{error_message_from_response[:100]}..."
             return f"抱歉，AI 服务返回了意外的响应格式({type(response).__name__})，可能是代理错误。详情: {detail}"

    # --- 错误处理 (保持不变) ---
    except AuthenticationError as e:
        print(f"!!! OpenAI 认证错误: {e} !!!")
        return "抱歉，连接 AI 服务时发生认证错误，请检查 API Key 是否正确或有权限。"
    except APIConnectionError as e:
        print(f"!!! OpenAI 连接错误: {e} !!!")
        return "抱歉，无法连接到 AI 服务，请检查网络连接或代理地址是否配置正确。"
    except RateLimitError as e:
        print(f"!!! OpenAI 速率限制错误: {e} !!!")
        return "抱歉，提问太快了，请稍后再试。"
    except Exception as e:
        # 打印完整错误栈信息
        print(f"!!! 调用 OpenAI API 时发生未知错误: {e} !!!")
        print(traceback.format_exc())
        return f"抱歉，与 AI 对话时遇到了未知问题 ({type(e).__name__})，请查看后台日志了解详情。"