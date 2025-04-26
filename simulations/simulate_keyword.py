# simulations/simulate_keyword.py

import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import logging
from typing import Optional, List, Dict, Any
import re # 导入正则表达式库用于关键词匹配
import time

# --- 配置 ---
CSV_FILE_PATH = 'products_export.csv' # 使用相同的 CSV 文件
# --- 模型配置 ---
# 使用与 RAG 模拟相同的对话模型
ACTIVE_CHAT_MODEL_NAME = 'gpt-4.1'
# --- 检索配置 ---
RETRIEVAL_TOP_N = 3 # 每次检索返回匹配度最高的 N 个结果
# --- 配置结束 ---

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- 日志设置结束 ---

# --- 库导入和客户端初始化 ---
OpenAI = None
APIError = None
RateLimitError = None
APIConnectionError = None
AuthenticationError = None
openai_client = None

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
except ImportError:
    logger.warning("openai library not found. Please install it: pip install openai")
# OpenAI client 在 load_environment_variables 中初始化
# --- 库导入和客户端初始化结束 ---

def load_environment_variables():
    """加载 .env 文件中的环境变量并初始化 OpenAI Client"""
    global openai_client
    try:
        env_path = find_dotenv()
        if not env_path:
             logger.warning(".env file not found.")
             return False
        load_dotenv(env_path, override=True)
        logger.info(f".env file loaded successfully from: {env_path}")

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")

        if not (api_key and api_base):
            logger.error("OpenAI API Key or Base URL missing in .env file.")
            return False
        if not OpenAI:
             logger.error("OpenAI library not loaded.")
             return False

        try:
             openai_client = OpenAI(api_key=api_key, base_url=api_base)
             logger.info("OpenAI client initialized successfully.")
             return True
        except Exception as client_err:
              logger.error(f"Failed to initialize OpenAI client: {client_err}", exc_info=True)
              openai_client = None
              return False
    except Exception as e:
        logger.error(f"Error loading .env file or initializing client: {e}", exc_info=True)
        return False

# --- 复用 load_product_data 函数 ---
def load_product_data(csv_relative_path: str) -> Optional[pd.DataFrame]:
    # ... (与 simulate_rag.py 中完全相同) ...
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        full_csv_path = os.path.join(project_root, csv_relative_path)
        logger.info(f"Attempting to load product data from: {full_csv_path}")

        df = pd.read_csv(full_csv_path)
        logger.info(f"Successfully loaded {len(df)} products from CSV.")
        logger.info(f"CSV Columns: {df.columns.tolist()}")

        required_cols = ['ID', 'Name', 'Description', 'Price', 'Stock']
        if not all(col in df.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in df.columns]
             logger.error(f"CSV missing one or more required columns: {missing_cols}")
             return None

        if 'Description' in df.columns:
             df['Description'] = df['Description'].fillna('')
             logger.info("Filled NaN values in 'Description' column with empty string.")
        else:
             logger.warning("'Description' column not found.")

        try:
            df['Price'] = pd.to_numeric(df['Price'])
            df['Stock'] = pd.to_numeric(df['Stock'])
            logger.info("Converted 'Price' and 'Stock' columns to numeric.")
        except Exception as e:
             logger.warning(f"Could not convert 'Price' or 'Stock' to numeric fully: {e}")

        return df
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {full_csv_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading or processing CSV file {full_csv_path}: {e}", exc_info=True)
        return None

# --- 新增函数：实现关键词检索逻辑 ---
def retrieve_keyword_docs(
    question_text: str,
    df: pd.DataFrame,
    top_n: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """根据问题文本中的关键词从 DataFrame 检索商品"""
    if df is None:
        logger.error("Product DataFrame is not available for keyword retrieval.")
        return None

    logger.info(f"\nRetrieving documents using keywords from: '{question_text}'")

    # 简单的关键词提取：这里我们直接使用整个问题作为关键词进行模糊匹配
    # 更复杂的可以分词、去停用词等
    keyword = question_text # 简化处理
    # 使用正则表达式进行不区分大小写的包含匹配
    # 同时搜索 Name 和 Description 字段
    try:
        # 确保 Name 和 Description 是字符串类型，以防万一
        df['Name'] = df['Name'].astype(str)
        df['Description'] = df['Description'].astype(str)

        # 使用 Pandas 的 str.contains 进行搜索，na=False 处理可能的NaN值
        name_matches = df[df['Name'].str.contains(keyword, case=False, na=False, regex=False)]
        desc_matches = df[df['Description'].str.contains(keyword, case=False, na=False, regex=False)]

        # 合并结果，并根据 ID 去重 (保留第一次出现的)
        combined_matches = pd.concat([name_matches, desc_matches]).drop_duplicates(subset=['ID'], keep='first')

        # 截取 top_n 个结果
        top_matches_df = combined_matches.head(top_n)

        # 格式化结果为字典列表
        retrieved_docs = []
        if not top_matches_df.empty:
            for index, row in top_matches_df.iterrows():
                # 注意：关键词检索没有直接的“距离”或“相似度”分数
                doc_info = {
                    "id": str(row['ID']), # 保持 ID 为字符串
                    "document": f"商品名称: {row['Name']}\n商品描述: {row['Description']}",
                    "metadata": { # 模拟 RAG 的 metadata 结构
                        "original_id": int(row['ID']),
                        "name": str(row['Name']),
                        "price": float(row['Price']) if pd.notna(row['Price']) else 0.0,
                        "stock": int(row['Stock']) if pd.notna(row['Stock']) else 0
                    },
                    "similarity": 1.0 # 简单标记为 1.0，表示匹配到
                }
                doc_info.update(doc_info.get("metadata", {})) # 提升元数据
                retrieved_docs.append(doc_info)

        logger.info(f"Retrieved {len(retrieved_docs)} documents based on keywords.")
        return retrieved_docs

    except Exception as e:
        logger.error(f"Error during keyword retrieval: {e}", exc_info=True)
        return None
# --- 关键词检索函数结束 ---


# --- 复用 build_rag_prompt 函数 (稍作调整或重命名) ---
def build_keyword_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """构建包含关键词检索到的上下文的 Prompt"""
    context = "--- 可能相关的商品信息 (基于关键词) ---\n"
    if not retrieved_docs:
        context += "(未找到包含关键词的商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1}:\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"
            # 描述可能很长，截断显示
            desc_full = doc.get('document', 'N/A').split('商品描述: ')[-1]
            context += f"  描述: {desc_full[:200]}...\n"
            context += f"  价格: {doc.get('price', 'N/A'):.2f}\n"
            context += f"  库存: {doc.get('stock', 'N/A')}\n\n"
            # 注意：这里的 similarity 只是标记匹配到，没有实际比较意义
    context += "--- 相关商品信息结束 ---\n\n"

    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面提供的“可能相关的商品信息”来回答用户的问题。\n"
        "这些信息是基于用户问题中的关键词找到的，不一定是语义最相关的。\n"
        "请主要依据这些信息进行回答，保持回答简洁、相关。\n"
        "如果提供的信息不足以回答问题，请直接说明“根据关键词找到的信息，无法直接回答您的问题。”\n"
        "不要编造信息中不存在的内容。"
    )

    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    return final_prompt

# --- 复用 get_llm_rag_response 函数 (重命名为 get_llm_response) ---
def get_llm_response(prompt: str, chat_model_name: str) -> str:
    """使用指定的对话模型获取回答"""
    if not openai_client:
        return "错误：OpenAI 客户端未初始化。"
    if not chat_model_name:
        return "错误：未指定对话模型。"

    logger.info(f"Sending prompt to LLM ({chat_model_name})... Prompt length: {len(prompt)}")
    logger.debug(f"LLM Prompt (first 300 chars): {prompt[:300]}...")

    try:
        response = openai_client.chat.completions.create(
            model=chat_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({chat_model_name}) generated answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({chat_model_name}) returned unexpected response: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as e:
        logger.error(f"LLM ({chat_model_name}) Rate Limit Error: {e}. Waiting and retrying once...")
        try:
             time.sleep(5)
             response = openai_client.chat.completions.create(model=chat_model_name, messages=[{"role": "user", "content": prompt}], max_tokens=500, temperature=0.3)
             if response.choices and response.choices[0].message and response.choices[0].message.content: return response.choices[0].message.content.strip()
             else: return "抱歉，重试后 AI 模型仍返回意外格式。"
        except Exception as retry_e:
             logger.error(f"LLM ({chat_model_name}) Retry Error: {retry_e}")
             return f"抱歉，尝试从速率限制中恢复失败：{retry_e}"
    except (APIConnectionError, AuthenticationError) as e:
         logger.error(f"LLM ({chat_model_name}) API/Auth Error: {e}")
         return f"抱歉，连接 AI 模型服务或认证失败：{e}"
    except Exception as e:
        logger.error(f"LLM ({chat_model_name}) Unexpected Error: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误：{e}"


# --- 主执行流程 ---
if __name__ == "__main__":
    logger.info("--- Starting Keyword Search Simulation Script ---")

    # 1. 加载环境变量并初始化 Client
    if not load_environment_variables():
        logger.critical("Failed to load environment or initialize OpenAI client. Exiting.")
        exit()

    # 2. 加载商品数据
    product_df = load_product_data(CSV_FILE_PATH)

    keyword_results = {} # 存储最终结果

    if product_df is not None:
        logger.info("Product data loaded successfully.")

        # --- 定义与 RAG 相同的测试问题 ---
        logger.info("\n--- Defining Test Questions ---")
        test_questions = [
            "有没有适合跑步的鞋？",
            "哪款手机屏幕最大？",
            "推荐一款便宜的固态硬盘",
            "介绍一下 iPhone 14 Pro",
            "牙刷怎么卖？",
            "最舒服的鞋子是哪双？",
            "库存最多的商品是什么？"
        ]
        logger.info(f"Defined {len(test_questions)} test questions.")

        # --- 运行关键词搜索+LLM流程 ---
        logger.info("\n--- Running Keyword Search + LLM Pipeline ---")

        for question in test_questions:
            print(f"\n{'='*10} Processing Question: {question} {'='*10}")

            # Step 1: Retrieve using Keywords
            retrieved_docs = retrieve_keyword_docs(question, product_df, top_n=RETRIEVAL_TOP_N)

            if retrieved_docs is None:
                print("  Keyword retrieval failed.")
                keyword_results[question] = {"retrieved": "ERROR", "prompt": "N/A", "answer": "ERROR"}
                continue
            elif not retrieved_docs:
                 print("  No documents found matching keywords.")

            # Step 2: Build Prompt
            final_prompt = build_keyword_prompt(question, retrieved_docs if retrieved_docs else [])
            print("\n  --- Retrieved Context (Formatted for Prompt) ---")
            context_part = final_prompt.split("--- 可能相关的商品信息 (基于关键词) ---\n")[1].split("--- 相关商品信息结束 ---\n")[0]
            print(context_part.strip())
            print("  --- End Retrieved Context ---")

            # Step 3: Get LLM Response
            llm_answer = get_llm_response(final_prompt, ACTIVE_CHAT_MODEL_NAME)

            # Step 4: Record Result
            keyword_results[question] = {
                "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                "top_retrieved_names": [doc.get('name') for doc in retrieved_docs[:RETRIEVAL_TOP_N]] if retrieved_docs else [],
                "answer": llm_answer
            }
            print(f"\n  --- LLM Answer ---")
            print(f"  {llm_answer}")
            print(f"  --- End LLM Answer ---")
            print(f"{'='*40}")
            time.sleep(1) # 稍微间隔

    else:
        logger.error("Failed to load product data. Exiting script.")

    logger.info("--- Keyword Search Simulation Script Finished ---")