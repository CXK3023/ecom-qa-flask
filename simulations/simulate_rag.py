# simulations/simulate_rag.py
# Version 4.4 Adaptation: Uses Persistent ChromaDB, reads new test_questions format, includes distractor_ids

import os
import sys
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import datetime
import decimal

# --- Database ---
import mysql.connector

# --- ChromaDB ---
try:
    import chromadb
    from chromadb.api.types import QueryResult
except ImportError:
    logging.error("chromadb library not found. Please install it: pip install chromadb")
    chromadb = None
    QueryResult = None # type: ignore

# --- OpenAI ---
try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
except ImportError:
    logging.warning("openai library not found. Please install it: pip install openai")
    OpenAI = None; APIError = Exception; RateLimitError = Exception; APIConnectionError = Exception; AuthenticationError = Exception

# --- Add project root to path to import app services ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
logger = logging.getLogger(__name__) # Define logger early
try:
    from app.services.embedding_service import get_active_embedding_model_info, generate_embedding, deserialize_embedding
    logger.info("Successfully imported app embedding services.")
    using_app_embedding_service = True
except ImportError as e:
    logger.warning(f"Could not import app embedding services: {e}. Using fallback embedding generation.", exc_info=True)
    using_app_embedding_service = False
    # 定义回退函数 (与上一个脚本保持一致)
    get_active_embedding_model_info = lambda: (os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "doubao-embedding-large-text"), "remote_api_fallback")
    def generate_embedding(text: str) -> Optional[bytes]:
         logger.warning("Using fallback generate_embedding. Ensure script's OpenAI client is initialized.")
         if not script_openai_client: logger.error("Fallback generate_embedding failed: script_openai_client not initialized."); return None
         try:
             model_name, _ = get_active_embedding_model_info()
             response = script_openai_client.embeddings.create(model=model_name, input=[text])
             if response.data and response.data[0].embedding:
                 vector = np.array(response.data[0].embedding); import pickle; return pickle.dumps(vector)
             return None
         except Exception as emb_err: logger.error(f"Fallback embedding generation failed: {emb_err}"); return None
    def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
         import pickle;
         if not serialized_embedding: return None
         try: vector = pickle.loads(serialized_embedding); return vector if isinstance(vector, np.ndarray) else None
         except Exception as des_err: logger.error(f"Fallback deserializing failed: {des_err}"); return None

# --- Configuration ---
# <<< MODIFIED: Point to the new test questions file >>>
TEST_QUESTIONS_FILE = 'simulations/test_questions_final.json' # <-- 使用新的文件名
# <<< MODIFIED: Suggest adding a version suffix to the results file >>>
RESULTS_FILE = 'simulations/rag_simulation_results_v4.4.json' # <-- 添加版本后缀
# --- Model Configuration ---
ACTIVE_CHAT_MODEL_NAME = "gpt-4.1" # Default, overridden by env var
# --- ChromaDB Configuration ---
CHROMA_DB_PATH = "./chroma_data_rag_v4_4" # <-- 使用新的持久化路径以隔离
CHROMA_COLLECTION_NAME = "products_rag_persistent_v4_4" # 使用新的集合名称
# --- RAG Configuration ---
RETRIEVAL_TOP_N = 3
# --- Configuration End ---

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
# --- Logging Setup End ---

# --- Global Variables ---
script_openai_client: Optional[OpenAI] = None
db_connection = None
db_cursor = None
# --- Global Variables End ---


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal): return float(obj)
        # 不需要处理 bytes 和 ndarray，因为它们不应出现在最终的 JSON 输出中
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    """加载环境变量并初始化 OpenAI 客户端"""
    global script_openai_client, ACTIVE_CHAT_MODEL_NAME
    try:
        env_path = find_dotenv()
        if not env_path: logger.warning(".env file not found.")
        else: load_dotenv(env_path, override=True); logger.info(f".env file loaded from: {env_path}")

        if OpenAI:
            api_key = os.getenv("OPENAI_API_KEY"); api_base = os.getenv("OPENAI_API_BASE")
            if not (api_key and api_base): logger.error("OpenAI API Key or Base URL missing in .env. Cannot initialize client."); return False
            try:
                script_openai_client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("Script's OpenAI client initialized successfully.")
                ACTIVE_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4.1")
                logger.info(f"Using Chat Model (from env or default): {ACTIVE_CHAT_MODEL_NAME}")
                return True
            except Exception as client_err: logger.error(f"Failed to initialize script's OpenAI client: {client_err}", exc_info=True); script_openai_client = None; return False
        else: logger.error("OpenAI library not loaded. Cannot initialize client."); return False
    except Exception as e: logger.error(f"Error loading .env file or initializing client: {e}", exc_info=True); return False

def get_db_connection():
    """建立数据库连接"""
    global db_connection, db_cursor
    if db_connection and db_connection.is_connected(): return db_connection, db_cursor
    db_url = os.getenv('DATABASE_URL')
    if not db_url: logger.critical("DATABASE_URL not found."); return None, None
    try:
        from urllib.parse import urlparse; parsed_url = urlparse(db_url)
        db_user = parsed_url.username; db_password = parsed_url.password; db_host = parsed_url.hostname
        db_port = parsed_url.port or 3306 # Default MySQL port
        db_name = parsed_url.path.lstrip('/')
        logger.info(f"Connecting to DB: {db_host}:{db_port}/{db_name} as {db_user}")
        db_connection = mysql.connector.connect(host=db_host, port=db_port, user=db_user, password=db_password, database=db_name)
        db_cursor = db_connection.cursor(dictionary=True)
        logger.info("Database connection established.")
        return db_connection, db_cursor
    except mysql.connector.Error as err: logger.error(f"Database connection failed: {err}", exc_info=True); db_connection = None; db_cursor = None; return None, None
    except Exception as e: logger.error(f"Unexpected error during DB connection: {e}", exc_info=True); db_connection = None; db_cursor = None; return None, None

def close_db_connection():
    """关闭数据库连接"""
    global db_connection, db_cursor
    if db_cursor: db_cursor.close(); logger.info("DB cursor closed.")
    if db_connection and db_connection.is_connected(): db_connection.close(); logger.info("DB connection closed.")
    db_cursor = None; db_connection = None

def fetch_products_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """从数据库获取商品基础信息"""
    conn, cursor = get_db_connection()
    if not conn or not cursor: return []
    products = []
    try:
        # 只获取 RAG 需要的基础字段
        query = "SELECT id, name, description, price, stock FROM product WHERE status = 'active'"
        if limit: query += f" LIMIT {limit}"
        cursor.execute(query)
        products = cursor.fetchall()
        logger.info(f"Fetched {len(products)} active products from database for RAG.")
    except mysql.connector.Error as err: logger.error(f"Error fetching products for RAG: {err}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error fetching products for RAG: {e}", exc_info=True)

    # 清理 None 值
    for p in products:
        if p.get('name') is None: p['name'] = ''
        if p.get('description') is None: p['description'] = ''
        # price 和 stock 在存入 ChromaDB metadata 时会处理
    return products

def generate_single_embedding_wrapper(text: str) -> Optional[List[float]]:
    """封装 embedding 生成和反序列化为 list"""
    if not text: return None
    serialized_vector = generate_embedding(text) # bytes
    if serialized_vector:
        vector_np = deserialize_embedding(serialized_vector) # ndarray
        if vector_np is not None:
            return vector_np.tolist() # ChromaDB 需要 list
    return None

def setup_chroma_vector_store(products: List[Dict[str, Any]]) -> Optional[chromadb.Collection]:
    """设置 ChromaDB 持久化存储。如果集合存在则加载，否则创建并生成 embedding。"""
    if not chromadb: logger.error("chromadb library is not available."); return None

    logger.info(f"Setting up ChromaDB persistent store at: {CHROMA_DB_PATH}")
    global simulation_results # Access global dict to store metadata
    if 'simulation_metadata' not in simulation_results: simulation_results['simulation_metadata'] = {}

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # --- 尝试获取现有集合 ---
        try:
            collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            logger.info(f"Loaded existing ChromaDB collection: '{CHROMA_COLLECTION_NAME}'")
            existing_count = collection.count()
            logger.info(f"Existing collection has {existing_count} items.")

            # --- 可以在这里添加检查，看是否需要基于当前 DB 数据更新索引 ---
            # --- （当前实现：不自动更新，如果需要重建请手动删除 CHROMA_DB_PATH）---
            db_count = len(products)
            if abs(existing_count - db_count) > max(10, db_count * 0.1): # 允许10%或10个的差异
                 logger.warning(f"Significant difference between items in existing ChromaDB ({existing_count}) "
                                f"and current DB product count ({db_count}). Index might be stale.")

            # 记录加载信息
            simulation_results['simulation_metadata']['indexing_action'] = f"Loaded Existing ({existing_count} items)"
            return collection

        except Exception as e: # 通常是 CollectionNotFoundError
            logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' not found or error getting it ({type(e).__name__}). Will create a new one.")

            # --- 创建新集合 ---
            collection = chroma_client.get_or_create_collection(
                CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"} # 指定余弦距离
            )
            logger.info(f"Created new ChromaDB collection: '{CHROMA_COLLECTION_NAME}' with cosine distance.")

            if not products: logger.error("No products fetched from DB, cannot populate new collection."); return None

            ids_to_add, embeddings_to_add, documents_to_add, metadatas_to_add = [], [], [], []
            failed_embeddings = 0
            start_time = time.time()

            logger.info("Generating embeddings and preparing data for new ChromaDB collection...")
            for product in tqdm(products, desc="Processing Products for ChromaDB"):
                product_id_str = str(product['id']) # ChromaDB ID 必须是字符串
                name_str = str(product.get('name', ''))
                desc_str = str(product.get('description', ''))
                # 构建用于 embedding 的文本
                text_to_embed = f"商品名称: {name_str}\n商品描述: {desc_str}"

                embedding_list = generate_single_embedding_wrapper(text_to_embed) # 获取 list[float]

                if embedding_list is None:
                    logger.warning(f"Skipping product ID {product_id_str} due to embedding generation failure.")
                    failed_embeddings += 1
                    continue

                ids_to_add.append(product_id_str)
                embeddings_to_add.append(embedding_list)
                documents_to_add.append(text_to_embed) # 可以存储用于调试或未来可能的重排序

                # 创建元数据，确保类型正确
                metadata = {
                    "original_id": int(product['id']), # 存储原始数据库 ID
                    "name": name_str,
                    # 处理 price 和 stock 的 None 值，转换为合适的类型
                    "price": float(product.get('price', 0.0) or 0.0),
                    "stock": int(product.get('stock', 0) or 0)
                }
                metadatas_to_add.append(metadata)

            if not ids_to_add:
                logger.error("No valid data to add to ChromaDB after embedding generation.")
                # 可选：删除空集合
                # try: chroma_client.delete_collection(CHROMA_COLLECTION_NAME) except: pass
                return None

            logger.info(f"Adding {len(ids_to_add)} items to ChromaDB collection (Embeddings failed for {failed_embeddings} items)...")
            # 分批添加以防数据量过大
            batch_size = 2000 # 根据内存调整
            for i in range(0, len(ids_to_add), batch_size):
                 end_index = min(i + batch_size, len(ids_to_add))
                 logger.info(f"Adding batch {i//batch_size + 1}: items {i} to {end_index-1}")
                 collection.add(
                     ids=ids_to_add[i:end_index],
                     embeddings=embeddings_to_add[i:end_index],
                     documents=documents_to_add[i:end_index],
                     metadatas=metadatas_to_add[i:end_index]
                 )
                 time.sleep(0.1) # 短暂休眠，避免可能的IO瓶颈

            end_time = time.time()
            indexing_time = end_time - start_time
            logger.info(f"Finished adding data. Indexing time: {indexing_time:.2f} seconds.")

            item_count = collection.count()
            logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' now contains {item_count} items.")

            # 记录索引信息
            simulation_results['simulation_metadata']['indexing_action'] = f"Created New ({item_count} items indexed)"
            simulation_results['simulation_metadata']['indexing_time_seconds'] = round(indexing_time, 2)
            simulation_results['simulation_metadata']['failed_embeddings_during_indexing'] = failed_embeddings

            return collection

    except Exception as e:
        logger.error(f"Fatal error during ChromaDB setup: {e}", exc_info=True)
        return None

def retrieve_relevant_docs(
    question_text: str, collection: chromadb.Collection, top_n: int = 3
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[float]]]:
    """使用 ChromaDB 检索相关文档"""
    if not collection: logger.error("ChromaDB collection is unavailable."); return None, None
    logger.info(f"Retrieving documents for question: '{question_text[:50]}...'")

    question_embedding = generate_single_embedding_wrapper(question_text) # list[float]
    if question_embedding is None:
        logger.error("Failed to generate question embedding for retrieval.")
        return None, None

    try:
        # 使用 question_embedding 查询
        results: Optional[QueryResult] = collection.query(
            query_embeddings=[question_embedding], # 需要 list of lists
            n_results=top_n,
            include=['metadatas', 'distances'] # 获取元数据和距离
        )
        logger.debug(f"ChromaDB query raw results: {results}")

        retrieved_docs = []
        # 解析结果 (ChromaDB 返回的结构可能嵌套一层 list)
        if results and results.get('ids') and results['ids'] and results['ids'][0]:
            ids_list = results['ids'][0]
            metadatas_list = results.get('metadatas', [[]])[0]
            distances_list = results.get('distances', [[]])[0]

            if len(ids_list) != len(metadatas_list) or len(ids_list) != len(distances_list):
                 logger.warning("ChromaDB query result lists length mismatch. Using minimum length.")
                 min_len = min(len(ids_list), len(metadatas_list), len(distances_list))
                 ids_list = ids_list[:min_len]
                 metadatas_list = metadatas_list[:min_len]
                 distances_list = distances_list[:min_len]


            for i in range(len(ids_list)):
                chroma_id = ids_list[i]
                metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                distance = distances_list[i] if distances_list and i < len(distances_list) and distances_list[i] is not None else -1.0
                # 余弦距离转相似度 (1 - distance)
                similarity = 1.0 - distance if distance >= 0 else 0.0 # 处理无效距离

                # 从 metadata 中提取信息
                doc_info = {
                    "chroma_id": chroma_id, # ChromaDB 内部 ID (字符串)
                    "metadata": metadata, # 原始元数据
                    "distance": distance,
                    "similarity": similarity,
                    # 从元数据获取原始数据库 ID 和其他信息
                    "id": metadata.get("original_id"), # 原始 DB ID
                    "name": metadata.get("name", "N/A"),
                    "price": metadata.get("price", "N/A"),
                    "stock": metadata.get("stock", "N/A"),
                }
                retrieved_docs.append(doc_info)

            logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB.")
            # 按相似度降序排序 (ChromaDB 默认按距离升序，所以转换后已经是降序了，但以防万一)
            retrieved_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return retrieved_docs, question_embedding # 返回文档列表和问题 embedding
        else:
            logger.info("No relevant documents found by ChromaDB query.")
            return [], question_embedding # 未找到，返回空列表

    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        return None, question_embedding # 查询出错

def build_rag_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """构建 RAG 的 Prompt"""
    context = "--- 相关商品信息 (来自向量数据库检索) ---\n"
    if not retrieved_docs:
        context += "(未找到相关商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1} (ID: {doc.get('id', 'N/A')}):\n" # 使用原始 DB ID
            context += f"  名称: {doc.get('name', 'N/A')}\n"

            # 价格格式化
            price_val = doc.get('price')
            if isinstance(price_val, (int, float, decimal.Decimal)):
                 try: price_str = f"{price_val:.2f}"
                 except TypeError: price_str = str(price_val)
            else: price_str = str(price_val) if price_val is not None else "N/A"
            context += f"  价格: {price_str}\n"

            context += f"  库存: {doc.get('stock', 'N/A')}\n"
            context += f"  (检索相似度: {doc.get('similarity', 0.0):.4f})\n\n" # 显示相似度
    context += "--- 相关商品信息结束 ---\n\n"

    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面通过“向量数据库检索”找到的“相关商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，要求简洁、准确、相关。\n"
        "如果提供的信息包含了用户提问的商品，请优先基于这些信息回答。\n"
        "如果提供的信息不足以回答，或者信息与问题无关，请直接说明“根据我找到的相关信息，无法直接回答您的问题。”或类似措辞。\n"
        "禁止编造信息中不存在的内容或价格。"
    )

    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    logger.debug(f"Generated RAG Prompt (first 300 chars): {final_prompt[:300]}...")
    return final_prompt

def get_llm_rag_response(prompt: str) -> str:
    """调用 LLM 获取 RAG 回答"""
    if not script_openai_client: return "错误：脚本的 OpenAI 客户端未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：无法确定活动的对话模型。"
    logger.info(f"Sending RAG prompt to LLM ({ACTIVE_CHAT_MODEL_NAME}). Length: {len(prompt)}")
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.2,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) generated RAG answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) returned an unexpected RAG response format: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as rle:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Rate Limit Error for RAG: {rle}. Waiting 5s...")
        time.sleep(5); return f"抱歉，已达到 API 速率限制，请稍后重试。 ({rle})"
    except (APIConnectionError, APIError, AuthenticationError) as apie:
         logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) API Error for RAG ({type(apie).__name__}): {apie}")
         return f"抱歉，与 AI 模型服务通信时出错 ({type(apie).__name__})。"
    except Exception as e:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Unexpected Error during RAG API call: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误。"

# <<< MODIFIED: Reuse the updated load_test_questions function from vector script >>>
# <<< (Ensure the function definition is identical to the one provided in the previous step) >>>
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
                 if not isinstance(q_data['ground_truth_ids'], list): logger.warning(f"Q ID {q_data.get('question_id', 'N/A')} has non-list 'ground_truth_ids'. Skipping."); continue
                 if not isinstance(q_data['distractor_ids'], list): logger.warning(f"Q ID {q_data.get('question_id', 'N/A')} has non-list 'distractor_ids'. Skipping."); continue
                 if not isinstance(q_data['ground_truth_answer_points'], list): logger.warning(f"Q ID {q_data.get('question_id', 'N/A')} has non-list 'ground_truth_answer_points'. Skipping."); continue
                 valid_questions.append(q_data)
            else: logger.error(f"Question #{i+1} in {full_path} is missing required keys ({required_keys}) or is not a dict. Skipping.")

        logger.info(f"Loaded {len(valid_questions)} valid questions after format check.")
        return valid_questions
    except FileNotFoundError: logger.error(f"Test questions file not found at {full_path}"); return []
    except json.JSONDecodeError as jde: logger.error(f"Error decoding JSON from {full_path}: {jde}", exc_info=True); return []
    except Exception as e: logger.error(f"Unexpected error loading test questions from {full_path}: {e}", exc_info=True); return []


# --- Main Execution ---
# <<< MODIFIED: Update metadata and result structure slightly >>>
simulation_results: Dict[str, Any] = {
    "simulation_metadata": {
        "script_name": os.path.basename(__file__),
        "script_version": "4.4-adaptation", # Added version
        "start_time": datetime.datetime.now().isoformat(),
        "search_method": "RAG (ChromaDB Persistent)",
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "chroma_db_path": CHROMA_DB_PATH,
        "retrieval_top_n": RETRIEVAL_TOP_N,
        "test_questions_file": TEST_QUESTIONS_FILE, # Record questions file
        # Indexing metadata will be added during setup_chroma_vector_store
    },
    "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting RAG Simulation Script ({os.path.basename(__file__)} - V4.4 Adaptation) ---")
    run_start_time = time.time()

    # 1. Load Environment & Client
    if not load_environment_variables():
        logger.critical("Failed to load environment or initialize OpenAI client. Exiting.")
        sys.exit(1)

    # Record model info
    emb_model, emb_method = get_active_embedding_model_info()
    simulation_results["simulation_metadata"]["embedding_model"] = f"{emb_model} ({emb_method})"
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME
    logger.info(f"Using Embedding Model: {emb_model} ({emb_method})")
    logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")

    # 2. Connect to DB & Fetch Products
    conn, cursor = get_db_connection()
    if not conn: logger.critical("Database connection failed. Exiting."); sys.exit(1)

    products = fetch_products_from_db() # Fetch base product info
    if not products:
        logger.warning("No active products fetched from database. RAG index might be empty or stale if creating new.")
        # Allow continuing if index might already exist
    simulation_results["simulation_metadata"]["total_products_fetched_from_db"] = len(products)

    # 3. Setup ChromaDB (Loads existing or Creates new & Indexes)
    product_collection = setup_chroma_vector_store(products)
    if not product_collection:
        logger.critical("Failed to setup ChromaDB vector store. Exiting.")
        close_db_connection()
        sys.exit(1)
    # Indexing metadata is added within setup_chroma_vector_store

    # 4. Load Test Questions
    test_questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not test_questions:
        logger.critical(f"Failed to load valid test questions from {TEST_QUESTIONS_FILE}. Exiting.")
        close_db_connection()
        sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions_processed"] = len(test_questions)

    # 5. Run RAG Pipeline for each question
    logger.info("\n--- Running RAG Pipeline for Test Questions ---")
    question_results_list = []
    total_retrieval_time_ms = 0
    total_llm_time_ms = 0

    for question_data in tqdm(test_questions, desc="Processing Questions (RAG)"):
        # <<< MODIFIED: Unpack new fields >>>
        q_id = question_data['question_id']
        q_text = question_data['question']
        gt_ids = question_data['ground_truth_ids']
        distractor_ids = question_data['distractor_ids'] # Extract distractors
        gt_answer_points = question_data['ground_truth_answer_points']
        # <<< END MODIFIED >>>

        logger.info(f"\nProcessing Question ID: {q_id} - '{q_text[:60]}...'")

        # --- Retrieval Phase ---
        retrieval_start_time = time.perf_counter()
        retrieved_docs, q_embedding = retrieve_relevant_docs(
            q_text, product_collection, top_n=RETRIEVAL_TOP_N
        )
        retrieval_end_time = time.perf_counter()
        retrieval_time_ms = (retrieval_end_time - retrieval_start_time) * 1000
        total_retrieval_time_ms += retrieval_time_ms

        # Prepare result entry structure early
        result_entry = {
            "question_id": q_id,
            "question": q_text,
            "ground_truth_ids": gt_ids,
            "distractor_ids": distractor_ids, # <<< MODIFIED: Include distractor_ids >>>
            "ground_truth_answer_points": gt_answer_points,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "llm_time_ms": 0,
            "retrieved_ids": [],
            "retrieved_details": [],
            "answer": "",
            "status": "Processing"
        }

        if retrieved_docs is None:
             logger.error(f"Retrieval failed for question ID {q_id}.")
             result_entry["status"] = "Retrieval Error"
             result_entry["answer"] = "ERROR: RAG retrieval phase failed."
             question_results_list.append(result_entry)
             continue # Skip to next question

        # Extract IDs and details from retrieved docs
        # Use the original DB ID ('id' key from metadata) for consistency
        retrieved_product_ids = [doc.get('id') for doc in retrieved_docs if doc.get('id') is not None]
        result_entry["retrieved_ids"] = retrieved_product_ids
        result_entry["retrieved_details"] = retrieved_docs # Store full details including similarity, distance, metadata

        logger.info(f"Retrieved {len(retrieved_docs)} docs. IDs: {retrieved_product_ids}. Time: {retrieval_time_ms:.2f} ms.")

        # --- LLM Phase ---
        final_prompt = build_rag_prompt(q_text, retrieved_docs)
        llm_start_time = time.perf_counter()
        llm_answer = get_llm_rag_response(final_prompt)
        llm_end_time = time.perf_counter()
        llm_time_ms = (llm_end_time - llm_start_time) * 1000
        total_llm_time_ms += llm_time_ms

        result_entry["llm_time_ms"] = round(llm_time_ms, 2)
        result_entry["answer"] = llm_answer
        result_entry["status"] = "Success"

        logger.info(f"LLM generation time: {llm_time_ms:.2f} ms.")

        question_results_list.append(result_entry)

        # Delay
        time.sleep(0.5) # RAG involves embedding + query + LLM, slightly longer delay

    # 6. Finalize and Save Results
    simulation_results["results"] = question_results_list

    # Calculate averages
    num_questions = len(test_questions)
    avg_retrieval_time = round(total_retrieval_time_ms / num_questions, 2) if num_questions > 0 else 0
    avg_llm_time = round(total_llm_time_ms / num_questions, 2) if num_questions > 0 else 0
    simulation_results["simulation_metadata"]["average_retrieval_time_ms"] = avg_retrieval_time
    simulation_results["simulation_metadata"]["average_llm_time_ms"] = avg_llm_time


    run_end_time = time.time()
    total_run_duration = run_end_time - run_start_time
    simulation_results["simulation_metadata"]["end_time"] = datetime.datetime.now().isoformat()
    simulation_results["simulation_metadata"]["total_duration_seconds"] = round(total_run_duration, 2)

    logger.info(f"\n--- RAG Simulation Finished ---")
    logger.info(f"Total execution time: {total_run_duration:.2f} seconds.")
    logger.info(f"Average Retrieval Time: {avg_retrieval_time:.2f} ms per question.")
    logger.info(f"Average LLM Time: {avg_llm_time:.2f} ms per question.")

    # Save results
    results_full_path = os.path.join(project_root, RESULTS_FILE)
    try:
        logger.info(f"Saving RAG results to {results_full_path}...")
        with open(results_full_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info(f"RAG results successfully saved to {results_full_path}")
    except Exception as e:
        logger.error(f"Error writing RAG results to {results_full_path}: {e}", exc_info=True)

    # 7. Close Database Connection
    close_db_connection()

    logger.info(f"--- RAG Simulation Script Completed ({os.path.basename(__file__)}) ---")