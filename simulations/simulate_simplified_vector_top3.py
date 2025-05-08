# simulations/simulate_simplified_vector.py
# Version 4.4 Adaptation: Reads new test_questions format, includes distractor_ids

import os
import sys
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import datetime
import decimal

# --- Database ---
import mysql.connector

# --- OpenAI ---
try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
except ImportError:
    logging.warning("openai library not found. Please install it: pip install openai")
    OpenAI = None; APIError = Exception; RateLimitError = Exception; APIConnectionError = Exception; AuthenticationError = Exception

# --- Add project root to path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
logger = logging.getLogger(__name__)
try:
    # 尝试导入应用服务，如果失败则使用回退逻辑
    from app.services.embedding_service import get_active_embedding_model_info, generate_embedding, deserialize_embedding, calculate_cosine_similarity
    logger.info("Successfully imported app embedding services.")
except ImportError as e:
    logger.error(f"Could not import app services: {e}. Using fallback.", exc_info=True)
    # 定义回退函数
    get_active_embedding_model_info = lambda: (os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "doubao-embedding-large-text"), "remote_api_fallback")
    def generate_embedding(text: str) -> Optional[bytes]:
         logger.warning("Using fallback generate_embedding."); import pickle
         if not script_openai_client: logger.error("Fallback failed: OpenAI client not initialized."); return None
         try:
             model_name, _ = get_active_embedding_model_info()
             response = script_openai_client.embeddings.create(model=model_name, input=[text])
         except Exception as emb_err: logger.error(f"Fallback embedding generation failed: {emb_err}"); return None
         if response.data and response.data[0].embedding:
             vector = np.array(response.data[0].embedding); return pickle.dumps(vector)
         return None
    def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
         import pickle;
         if not serialized_embedding: return None
         try: vector = pickle.loads(serialized_embedding); return vector if isinstance(vector, np.ndarray) else None
         except Exception as des_err: logger.error(f"Fallback deserializing failed: {des_err}"); return None
    def calculate_cosine_similarity(vec1, vec2):
        if vec1 is None or vec2 is None: return -1.0
        try:
             # 确保 vec1 和 vec2 是 NumPy 数组
             if not isinstance(vec1, np.ndarray): vec1 = np.array(vec1)
             if not isinstance(vec2, np.ndarray): vec2 = np.array(vec2)
             # 如果向量是空的，返回-1
             if vec1.size == 0 or vec2.size == 0: return -1.0
             # 计算余弦相似度
             sim = 1.0 - cdist(vec1.reshape(1,-1), vec2.reshape(1,-1), 'cosine')[0,0];
             return max(-1.0, min(1.0, sim)) if not np.isnan(sim) else -1.0
        except ValueError as ve: # 捕捉维度不匹配等错误
            logger.error(f"Cosine similarity calculation error: {ve}. Vec1 shape: {vec1.shape}, Vec2 shape: {vec2.shape}")
            return -1.0
        except Exception as e:
            logger.error(f"Unexpected error in cosine similarity: {e}")
            return -1.0


# --- Configuration ---
# <<< MODIFIED: Point to the new test questions file >>>
TEST_QUESTIONS_FILE = 'simulations/test_questions_final.json' # <-- 使用新的文件名
# <<< MODIFIED: Suggest adding a version suffix to the results file >>>
RESULTS_FILE = 'simulations/simplified_vector_simulation_results_top3.json' # <-- 添加版本后缀
ACTIVE_CHAT_MODEL_NAME = "gpt-4.1" # 默认值，会被环境变量覆盖
RETRIEVAL_TOP_N = 3
# --- Config End ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
script_openai_client: Optional[OpenAI] = None
db_connection = None; db_cursor = None

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, bytes): return '<<bytes>>' # 不序列化 bytes
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    global script_openai_client, ACTIVE_CHAT_MODEL_NAME
    try:
        env_path = find_dotenv();
        if env_path: load_dotenv(env_path, override=True); logger.info(f".env loaded: {env_path}")
        else: logger.warning(".env not found.")
        if OpenAI:
            api_key = os.getenv("OPENAI_API_KEY"); api_base = os.getenv("OPENAI_API_BASE")
            if not (api_key and api_base): logger.error("OpenAI API Key or Base URL missing in .env. Cannot initialize client."); return False
            try:
                script_openai_client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("Script's OpenAI client initialized successfully.")
                ACTIVE_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4.1") # 从环境变量读取或使用默认值
                logger.info(f"Using Chat Model (from env or default): {ACTIVE_CHAT_MODEL_NAME}")
                return True
            except Exception as client_err: logger.error(f"Failed to initialize script's OpenAI client: {client_err}", exc_info=True); script_openai_client = None; return False
        else: logger.error("OpenAI library not loaded. Cannot initialize client."); return False
    except Exception as e: logger.error(f"Error loading .env file or initializing client: {e}", exc_info=True); return False

def get_db_connection():
    global db_connection, db_cursor
    if db_connection and db_connection.is_connected(): return db_connection, db_cursor
    db_url = os.getenv('DATABASE_URL');
    if not db_url: logger.critical("DATABASE_URL not found in environment variables."); return None, None
    try:
        from urllib.parse import urlparse; p = urlparse(db_url)
        logger.info(f"Connecting to DB: {p.hostname}:{p.port or 3306}/{p.path.lstrip('/')} as {p.username}") # 添加默认端口处理
        db_connection = mysql.connector.connect(
            host=p.hostname, port=p.port or 3306, user=p.username, password=p.password, database=p.path.lstrip('/')
        )
        db_cursor = db_connection.cursor(dictionary=True); logger.info("Database connection established.")
        return db_connection, db_cursor
    except mysql.connector.Error as err: logger.error(f"DB connection failed: {err}", exc_info=True); return None, None
    except Exception as e: logger.error(f"Unexpected error during DB connection: {e}", exc_info=True); return None, None


def close_db_connection():
    global db_connection, db_cursor
    if db_cursor: db_cursor.close(); logger.info("DB cursor closed.")
    if db_connection and db_connection.is_connected(): db_connection.close(); logger.info("DB connection closed.")
    db_cursor = None; db_connection = None

def fetch_products_with_embeddings(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """从数据库获取商品及其预计算的 embedding"""
    conn, cursor = get_db_connection();
    if not conn or not cursor: return []
    valid_products = []
    try:
        # 假设 embedding 字段存储了预计算的向量 (bytes)
        query = "SELECT id, name, description, price, stock, embedding FROM product WHERE status = 'active'"
        if limit: query += f" LIMIT {limit}"
        cursor.execute(query); products = cursor.fetchall();
        logger.info(f"Fetched {len(products)} raw products from database.")

        logger.info("Deserializing embeddings...")
        missing_embeddings, deserialization_errors = 0, 0
        for product in products:
            serialized_emb = product.get('embedding')
            if serialized_emb and isinstance(serialized_emb, bytes):
                try:
                    vector = deserialize_embedding(serialized_emb)
                    if vector is not None:
                        product['vector'] = vector # 将反序列化的向量存入字典
                        valid_products.append(product)
                    else:
                        deserialization_errors += 1
                        logger.warning(f"Deserialized embedding is None for product ID {product.get('id')}.")
                except Exception as e:
                    deserialization_errors += 1
                    logger.error(f"Failed to deserialize embedding for product ID {product.get('id')}: {e}")
            else:
                missing_embeddings += 1
                logger.warning(f"Missing or invalid embedding (not bytes) for product ID {product.get('id')}.")

        logger.info(f"Processed {len(valid_products)} products with valid embeddings. "
                    f"Missing/Invalid: {missing_embeddings}, Deserialization Errors: {deserialization_errors}.")
    except mysql.connector.Error as err:
        logger.error(f"Database error fetching products with embeddings: {err}", exc_info=True)
        valid_products = []
    except Exception as e:
        logger.error(f"Unexpected error fetching products with embeddings: {e}", exc_info=True)
        valid_products = []

    # 清理 None 值，防止后续处理出错
    for p in valid_products:
        if p.get('name') is None: p['name'] = ''
        if p.get('description') is None: p['description'] = ''
        # price 和 stock 在后续使用时通常会处理 None 或转换类型

    return valid_products

def generate_single_embedding_wrapper(text: str) -> Optional[np.ndarray]:
    """封装 embedding 生成和反序列化"""
    if not text: return None
    serialized_vector = generate_embedding(text) # 调用服务或回退逻辑生成 bytes
    if serialized_vector:
        return deserialize_embedding(serialized_vector) # 反序列化为 ndarray
    return None

def find_similar_products_simplified(
    question_text: str, products_with_vectors: List[Dict[str, Any]], top_n: int = 3
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray]]:
    """使用预计算的向量进行简化版相似度搜索"""
    if not products_with_vectors:
        logger.error("No products with vectors provided for search."); return None, None # 返回 None 表示失败

    logger.info(f"Performing simplified vector search for question: '{question_text[:50]}...'")
    question_vector = generate_single_embedding_wrapper(question_text)
    if question_vector is None:
        logger.error("Failed to generate embedding for the question."); return None, None # 失败

    try:
        # 提取所有有效的商品向量
        product_vectors = [p['vector'] for p in products_with_vectors if isinstance(p.get('vector'), np.ndarray)]
        if not product_vectors:
            logger.error("No valid product vectors found in the provided list."); return [], question_vector # 没有可比较的向量，返回空列表

        # 堆叠向量矩阵
        all_product_vectors_matrix = np.vstack(product_vectors)
        logger.debug(f"Product vectors matrix shape: {all_product_vectors_matrix.shape}")

        # 检查维度是否匹配
        if question_vector.shape[0] != all_product_vectors_matrix.shape[1]:
            logger.error(f"Dimension mismatch: Question vector ({question_vector.shape}) vs Product vectors ({all_product_vectors_matrix.shape})")
            return None, question_vector # 维度不匹配，搜索失败

    except Exception as e:
        logger.error(f"Error preparing vectors for distance calculation: {e}", exc_info=True)
        return None, question_vector # 准备阶段出错，失败

    try:
        logger.info("Calculating cosine distances...")
        # cdist 需要 2D 数组
        question_vector_2d = question_vector.reshape(1, -1)
        # 计算问题向量与所有商品向量的余弦距离
        distances = cdist(question_vector_2d, all_product_vectors_matrix, 'cosine')[0] # [0] 获取距离数组
        logger.info(f"Distances calculated. Shape: {distances.shape}")

    except ValueError as ve:
         logger.error(f"Error during cdist calculation (likely shape mismatch): {ve}", exc_info=True)
         return None, question_vector # 计算出错
    except Exception as e:
        logger.error(f"Unexpected error during distance calculation: {e}", exc_info=True)
        return None, question_vector # 计算出错

    try:
        # 获取距离最小的 top_n 个索引
        # 注意：argsort 默认升序，距离越小越相似
        sorted_indices = np.argsort(distances)
        top_n_indices = sorted_indices[:top_n]

        # 构建结果列表
        top_products = []
        original_indices_map = {i: p_idx for i, p_idx in enumerate(idx for idx, p in enumerate(products_with_vectors) if isinstance(p.get('vector'), np.ndarray))}

        for i in top_n_indices:
             original_product_index = original_indices_map.get(i)
             if original_product_index is None:
                  logger.warning(f"Could not map sorted index {i} back to original product list.")
                  continue

             product_dict = products_with_vectors[original_product_index].copy() # 复制商品信息
             distance = distances[i]
             similarity = 1.0 - distance # 转换为相似度
             product_dict['similarity'] = max(-1.0, min(1.0, similarity)) # 限制在 [-1, 1]
             product_dict['distance'] = distance
             top_products.append(product_dict)

        logger.info(f"Found top {len(top_products)} similar products based on vector distance.")
        return top_products, question_vector # 返回找到的商品和问题向量

    except Exception as e:
        logger.error(f"Error selecting top N products after distance calculation: {e}", exc_info=True)
        return None, question_vector # 选择 Top N 出错


def build_simplified_vector_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """构建用于 LLM 的 Prompt，包含检索到的商品信息"""
    context = "--- 相关商品信息 (基于向量相似度) ---\n"
    if not retrieved_docs:
        context += "(未找到向量相似的商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1} (ID: {doc.get('id', 'N/A')}):\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"

            # 价格格式化 (处理 Decimal, float, int, None, str)
            price_val = doc.get('price')
            if isinstance(price_val, (int, float, decimal.Decimal)):
                try:
                    price_str = f"{price_val:.2f}" # 格式化为两位小数
                except TypeError: # 处理 Decimal 可能的格式化问题
                    price_str = str(price_val)
            else:
                price_str = str(price_val) if price_val is not None else "N/A"
            context += f"  价格: {price_str}\n"

            context += f"  库存: {doc.get('stock', 'N/A')}\n"
            context += f"  (向量相似度: {doc.get('similarity', 0.0):.4f})\n\n" # 显示相似度
    context += "--- 相关商品信息结束 ---\n\n"

    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面通过“向量相似度”找到的“相关商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，要求简洁、准确、相关。\n"
        "如果提供的信息包含了用户提问的商品，请优先基于这些信息回答。\n"
        "如果提供的信息不足以回答，或者信息与问题无关，请直接说明“根据我找到的相关信息，无法直接回答您的问题。”或类似措辞。\n"
        "禁止编造信息中不存在的内容或价格。"
    )

    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    logger.debug(f"Generated Prompt (first 300 chars): {final_prompt[:300]}...")
    return final_prompt


def get_llm_simplified_vector_response(prompt: str) -> str:
    """调用 LLM 获取回答"""
    if not script_openai_client: return "错误：脚本的 OpenAI 客户端未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：无法确定活动的对话模型。"
    logger.info(f"Sending prompt to LLM ({ACTIVE_CHAT_MODEL_NAME}). Length: {len(prompt)}")
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, # 可根据需要调整
            temperature=0.2, # 较低的温度以获得更确定的回答
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) generated answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) returned an unexpected response format: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as rle:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Rate Limit Error: {rle}. Waiting 5s...")
        time.sleep(5); return f"抱歉，已达到 API 速率限制，请稍后重试。 ({rle})"
    except (APIConnectionError, APIError, AuthenticationError) as apie:
         logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) API Error ({type(apie).__name__}): {apie}")
         return f"抱歉，与 AI 模型服务通信时出错 ({type(apie).__name__})。"
    except Exception as e:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Unexpected Error during API call: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误。"

# <<< MODIFIED: Updated load_test_questions function >>>
def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load test questions from the new JSON format."""
    try:
        full_path = os.path.join(project_root, file_path)
        logger.info(f"Attempting to load test questions from: {full_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            questions_list = json.load(f)
        logger.info(f"Successfully loaded {len(questions_list)} questions from {full_path}")

        # Validate keys for the new format
        required_keys = ['question_id', 'question', 'ground_truth_ids', 'distractor_ids', 'ground_truth_answer_points']
        valid_questions = []
        for i, q_data in enumerate(questions_list):
            if isinstance(q_data, dict) and all(key in q_data for key in required_keys):
                # Basic type checks (can be expanded)
                if not isinstance(q_data['ground_truth_ids'], list):
                     logger.warning(f"Question #{i+1} (ID: {q_data.get('question_id', 'N/A')}) has non-list 'ground_truth_ids'. Skipping.")
                     continue
                if not isinstance(q_data['distractor_ids'], list):
                     logger.warning(f"Question #{i+1} (ID: {q_data.get('question_id', 'N/A')}) has non-list 'distractor_ids'. Skipping.")
                     continue
                if not isinstance(q_data['ground_truth_answer_points'], list):
                     logger.warning(f"Question #{i+1} (ID: {q_data.get('question_id', 'N/A')}) has non-list 'ground_truth_answer_points'. Skipping.")
                     continue
                valid_questions.append(q_data)
            else:
                logger.error(f"Question #{i+1} in {full_path} is missing required keys ({required_keys}) or is not a dictionary. Skipping.")

        logger.info(f"Loaded {len(valid_questions)} valid questions after format check.")
        return valid_questions
    except FileNotFoundError:
        logger.error(f"Test questions file not found at {full_path}")
        return []
    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from {full_path}: {jde}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading test questions from {full_path}: {e}", exc_info=True)
        return []
# <<< END MODIFIED function >>>


# --- Main Execution ---
# <<< MODIFIED: Update metadata and result structure slightly >>>
simulation_results: Dict[str, Any] = {
    "simulation_metadata": {
        "script_name": os.path.basename(__file__),
        "script_version": "4.4-adaptation", # Added version
        "start_time": datetime.datetime.now().isoformat(),
        "search_method": "Simplified Vector Search (DB Embeddings + cdist)",
        "retrieval_top_n": RETRIEVAL_TOP_N,
        "test_questions_file": TEST_QUESTIONS_FILE, # Record which questions file was used
    },
    "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting Script ({os.path.basename(__file__)}) ---")
    run_start_time = time.time()

    # 1. Load Environment & Initialize OpenAI Client
    if not load_environment_variables():
        logger.critical("Failed to load environment variables or initialize OpenAI client. Exiting.")
        sys.exit(1)

    # Record model info in metadata
    embedding_model_name, embedding_method = get_active_embedding_model_info()
    simulation_results["simulation_metadata"]["embedding_model"] = f"{embedding_model_name} ({embedding_method})"
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME
    logger.info(f"Using Embedding Model: {embedding_model_name} ({embedding_method})")
    logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")

    # 2. Connect to Database & Fetch Products with Embeddings
    conn, cursor = get_db_connection()
    if not conn:
        logger.critical("Database connection failed. Exiting.")
        sys.exit(1)

    products_with_vectors = fetch_products_with_embeddings() # Fetch products and their vectors
    if not products_with_vectors:
        # Decide if this is critical. Maybe allow running with 0 products?
        logger.error("No products with valid vectors fetched from the database. Cannot proceed with search.")
        close_db_connection()
        sys.exit(1) # Exit if no products to search
    simulation_results["simulation_metadata"]["total_products_fetched_with_vector"] = len(products_with_vectors)
    logger.info(f"{len(products_with_vectors)} product vectors loaded successfully.")

    # 3. Load Test Questions (using the modified function)
    test_questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not test_questions:
        logger.critical(f"Failed to load valid test questions from {TEST_QUESTIONS_FILE}. Exiting.")
        close_db_connection()
        sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions_processed"] = len(test_questions) # Record how many were processed

    # 4. Run Simulation Pipeline for each question
    logger.info("\n--- Running Simplified Vector Search Pipeline ---")
    question_results_list = []
    total_search_time_ms = 0
    total_llm_time_ms = 0

    for question_data in tqdm(test_questions, desc="Processing Questions (Vector Search)"):
        # <<< MODIFIED: Unpack new fields >>>
        q_id = question_data['question_id']
        q_text = question_data['question']
        gt_ids = question_data['ground_truth_ids']
        distractor_ids = question_data['distractor_ids'] # Extract distractor IDs
        gt_answer_points = question_data['ground_truth_answer_points']
        # <<< END MODIFIED >>>

        logger.info(f"\nProcessing Question ID: {q_id} - '{q_text[:60]}...'")

        # --- Search Phase ---
        search_start_time = time.perf_counter()
        # Pass the loaded products with vectors to the search function
        retrieved_products, _ = find_similar_products_simplified(
            q_text, products_with_vectors, top_n=RETRIEVAL_TOP_N
        )
        search_end_time = time.perf_counter()
        search_time_ms = (search_end_time - search_start_time) * 1000
        total_search_time_ms += search_time_ms

        result_entry = {
            "question_id": q_id,
            "question": q_text,
            "ground_truth_ids": gt_ids,
            "distractor_ids": distractor_ids, # <<< MODIFIED: Include distractor_ids >>>
            "ground_truth_answer_points": gt_answer_points,
            "search_time_ms": round(search_time_ms, 2),
            "llm_time_ms": 0, # Initialize LLM time
            "retrieved_ids": [],
            "retrieved_details": [],
            "answer": "",
            "status": "Processing"
        }

        if retrieved_products is None:
            logger.error(f"Vector search failed for question ID {q_id}.")
            result_entry["status"] = "Search Error"
            result_entry["answer"] = "ERROR: Vector search phase failed."
            question_results_list.append(result_entry)
            continue # Skip to next question if search fails

        # Extract retrieved IDs and details (remove vector before saving)
        retrieved_ids = [p.get('id') for p in retrieved_products if p.get('id') is not None]
        result_entry["retrieved_ids"] = retrieved_ids
        # Clean up details before saving (remove large vector data)
        cleaned_details = []
        for detail in retrieved_products:
             cleaned_copy = detail.copy()
             cleaned_copy.pop('vector', None) # Remove vector
             cleaned_copy.pop('embedding', None) # Remove raw embedding if present
             cleaned_details.append(cleaned_copy)
        result_entry["retrieved_details"] = cleaned_details

        logger.info(f"Retrieved {len(retrieved_products)} products. IDs: {retrieved_ids}. Search Time: {search_time_ms:.2f} ms.")

        # --- LLM Phase ---
        final_prompt = build_simplified_vector_prompt(q_text, retrieved_products)

        llm_start_time = time.perf_counter()
        llm_answer = get_llm_simplified_vector_response(final_prompt)
        llm_end_time = time.perf_counter()
        llm_time_ms = (llm_end_time - llm_start_time) * 1000
        total_llm_time_ms += llm_time_ms

        result_entry["llm_time_ms"] = round(llm_time_ms, 2)
        result_entry["answer"] = llm_answer
        result_entry["status"] = "Success" # Mark as success if LLM call returns

        logger.info(f"LLM generation time: {llm_time_ms:.2f} ms.")

        question_results_list.append(result_entry)

        # Optional delay between questions to avoid hitting rate limits too quickly
        time.sleep(0.3) # 300ms delay

    # 5. Finalize and Save Results
    simulation_results["results"] = question_results_list

    # Calculate averages
    num_questions = len(test_questions)
    avg_search_time = round(total_search_time_ms / num_questions, 2) if num_questions > 0 else 0
    avg_llm_time = round(total_llm_time_ms / num_questions, 2) if num_questions > 0 else 0
    simulation_results["simulation_metadata"]["average_search_time_ms"] = avg_search_time
    simulation_results["simulation_metadata"]["average_llm_time_ms"] = avg_llm_time

    run_end_time = time.time()
    total_run_duration = run_end_time - run_start_time
    simulation_results["simulation_metadata"]["end_time"] = datetime.datetime.now().isoformat()
    simulation_results["simulation_metadata"]["total_duration_seconds"] = round(total_run_duration, 2)

    logger.info(f"\n--- Simplified Vector Simulation Finished ---")
    logger.info(f"Total execution time: {total_run_duration:.2f} seconds.")
    logger.info(f"Average Search Time: {avg_search_time:.2f} ms per question.")
    logger.info(f"Average LLM Time: {avg_llm_time:.2f} ms per question.")

    # Save results to JSON file
    results_full_path = os.path.join(project_root, RESULTS_FILE)
    try:
        logger.info(f"Saving results to {results_full_path}...")
        with open(results_full_path, 'w', encoding='utf-8') as f:
            # Use DecimalEncoder to handle potential Decimal types from DB metadata
            json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info(f"Results successfully saved to {results_full_path}")
    except Exception as e:
        logger.error(f"Error writing results to {results_full_path}: {e}", exc_info=True)

    # 6. Close Database Connection
    close_db_connection()

    logger.info(f"--- Script Completed ({os.path.basename(__file__)}) ---")