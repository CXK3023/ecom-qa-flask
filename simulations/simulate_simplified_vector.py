# simulations/simulate_simplified_vector.py
# Version 3.1: Fixes ValueError in price formatting

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
    from app.services.embedding_service import get_active_embedding_model_info, generate_embedding, deserialize_embedding, calculate_cosine_similarity
    logger.info("Successfully imported app embedding services.")
except ImportError as e:
    logger.error(f"Could not import app services: {e}. Using fallback.", exc_info=True)
    get_active_embedding_model_info = lambda: (os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "doubao-embedding-large-text"), "remote_api")
    def generate_embedding(text: str) -> Optional[bytes]:
         logger.warning("Using fallback generate_embedding."); import pickle
         if not script_openai_client: logger.error("Fallback failed: client init."); return None
         try: model_name, _ = get_active_embedding_model_info(); response = script_openai_client.embeddings.create(model=model_name, input=[text])
         except Exception as emb_err: logger.error(f"Fallback embedding failed: {emb_err}"); return None
         if response.data and response.data[0].embedding: vector = np.array(response.data[0].embedding); return pickle.dumps(vector)
         return None
    def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
         import pickle;
         if not serialized_embedding: return None
         try: vector = pickle.loads(serialized_embedding); return vector if isinstance(vector, np.ndarray) else None
         except Exception as des_err: logger.error(f"Fallback deserializing failed: {des_err}"); return None
    def calculate_cosine_similarity(vec1, vec2):
        if vec1 is None or vec2 is None: return -1.0
        try: sim = 1.0 - cdist(vec1.reshape(1,-1), vec2.reshape(1,-1), 'cosine')[0,0]; return max(-1.0, min(1.0, sim)) if not np.isnan(sim) else -1.0
        except: return -1.0

# --- Configuration ---
TEST_QUESTIONS_FILE = 'simulations/test_questions.json'
RESULTS_FILE = 'simulations/simplified_vector_simulation_results.json'
ACTIVE_CHAT_MODEL_NAME = "gpt-4.1"
RETRIEVAL_TOP_N = 3
# --- Config End ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
script_openai_client: Optional[OpenAI] = None
db_connection = None; db_cursor = None

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, bytes): return '<<bytes>>'
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    global script_openai_client, ACTIVE_CHAT_MODEL_NAME
    try:
        env_path = find_dotenv();
        if env_path: load_dotenv(env_path, override=True); logger.info(f".env loaded: {env_path}")
        else: logger.warning(".env not found.")
        if OpenAI:
            api_key = os.getenv("OPENAI_API_KEY"); api_base = os.getenv("OPENAI_API_BASE")
            if not (api_key and api_base): logger.error("API Key/Base missing."); return False
            try: script_openai_client = OpenAI(api_key=api_key, base_url=api_base); logger.info("Script client initialized.")
            except Exception as e: logger.error(f"Client init failed: {e}"); return False
            ACTIVE_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4.1"); logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")
            return True
        else: logger.error("OpenAI lib not loaded."); return False
    except Exception as e: logger.error(f"Env loading failed: {e}"); return False

def get_db_connection():
    global db_connection, db_cursor
    if db_connection and db_connection.is_connected(): return db_connection, db_cursor
    db_url = os.getenv('DATABASE_URL');
    if not db_url: logger.critical("DATABASE_URL not found."); return None, None
    try:
        from urllib.parse import urlparse; p = urlparse(db_url)
        logger.info(f"Connecting DB: {p.hostname}:{p.port}/{p.path.lstrip('/')} as {p.username}")
        db_connection = mysql.connector.connect(host=p.hostname, port=p.port, user=p.username, password=p.password, database=p.path.lstrip('/'))
        db_cursor = db_connection.cursor(dictionary=True); logger.info("DB connection established.")
        return db_connection, db_cursor
    except Exception as e: logger.error(f"DB connection failed: {e}"); return None, None

def close_db_connection():
    global db_connection, db_cursor
    if db_cursor: db_cursor.close(); logger.info("DB cursor closed.")
    if db_connection and db_connection.is_connected(): db_connection.close(); logger.info("DB connection closed.")
    db_cursor = None; db_connection = None

def fetch_products_with_embeddings(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    conn, cursor = get_db_connection();
    if not conn or not cursor: return []
    valid_products = []
    try:
        query = "SELECT id, name, description, price, stock, embedding FROM product WHERE status = 'active'" + (f" LIMIT {limit}" if limit else "")
        cursor.execute(query); products = cursor.fetchall(); logger.info(f"Fetched {len(products)} raw products.")
        logger.info("Deserializing embeddings...")
        missing, errors = 0, 0
        for p in products:
            emb = p.get('embedding')
            if emb and isinstance(emb, bytes):
                try:
                    vec = deserialize_embedding(emb)
                    if vec is not None: p['vector'] = vec; valid_products.append(p)
                    else: errors += 1; logger.warning(f"Deserialize None for ID {p.get('id')}.")
                except Exception as e: errors += 1; logger.error(f"Deserialize error for ID {p.get('id')}: {e}")
            else: missing += 1; logger.warning(f"Missing/invalid embedding for ID {p.get('id')}.")
        logger.info(f"Processed {len(valid_products)} products with valid embeddings. Missing: {missing}, Errors: {errors}.")
    except Exception as e: logger.error(f"Fetching products failed: {e}"); valid_products = []
    for p in valid_products:
        if p.get('name') is None: p['name'] = ''
        if p.get('description') is None: p['description'] = ''
    return valid_products

def generate_single_embedding_wrapper(text: str) -> Optional[np.ndarray]:
    if not text: return None
    ser_vec = generate_embedding(text)
    return deserialize_embedding(ser_vec) if ser_vec else None

def find_similar_products_simplified(
    question_text: str, products_with_vectors: List[Dict[str, Any]], top_n: int = 3
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray]]:
    if not products_with_vectors: logger.error("No products for search."); return [], None
    logger.info(f"Simplified vector search for: '{question_text}'")
    q_vec = generate_single_embedding_wrapper(question_text)
    if q_vec is None: logger.error("Q-embedding failed."); return None, None
    try:
        valid_vectors = [p['vector'] for p in products_with_vectors if isinstance(p.get('vector'), np.ndarray)]
        if not valid_vectors: logger.error("No valid product vectors."); return [], q_vec
        all_prod_vecs = np.vstack(valid_vectors)
        logger.info(f"Product vectors shape: {all_prod_vecs.shape}")
        if q_vec.shape[0] != all_prod_vecs.shape[1]: logger.error(f"Dim mismatch: Q({q_vec.shape}) vs P({all_prod_vecs.shape})"); return None, q_vec
    except Exception as e: logger.error(f"Vector prep failed: {e}"); return None, q_vec
    try:
        logger.info("Calculating distances..."); q_vec_2d = q_vec.reshape(1, -1)
        distances = cdist(q_vec_2d, all_prod_vecs, 'cosine')[0]; logger.info("Distances calculated.")
    except Exception as e: logger.error(f"cdist failed: {e}"); return None, q_vec
    try:
        sorted_indices = np.argsort(distances); top_n_indices = sorted_indices[:top_n]
        top_products = []
        for i in top_n_indices:
            p_dict = products_with_vectors[i].copy(); dist = distances[i]
            sim = 1.0 - dist; p_dict['similarity'] = max(-1.0, min(1.0, sim))
            p_dict['distance'] = dist; top_products.append(p_dict)
        logger.info(f"Found top {len(top_products)} products."); return top_products, q_vec
    except Exception as e: logger.error(f"Top N selection failed: {e}"); return None, q_vec

# --- FIX: Corrected price formatting logic ---
def build_simplified_vector_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = "--- 相关商品信息 (基于向量相似度) ---\n"
    if not retrieved_docs:
        context += "(未找到向量相似的商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1} (ID: {doc.get('id', 'N/A')}):\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"

            # --- Corrected Price Formatting ---
            price_val = doc.get('price') # Get raw price value
            if isinstance(price_val, (int, float, decimal.Decimal)):
                # Only apply .2f if it's a number
                price_str = f"{price_val:.2f}"
            else:
                # Otherwise, convert to string (handles None, strings, etc.)
                price_str = str(price_val) if price_val is not None else "N/A"
            context += f"  价格: {price_str}\n"
            # --- End Corrected Price Formatting ---

            context += f"  库存: {doc.get('stock', 'N/A')}\n"
            context += f"  (向量相似度: {doc.get('similarity', 0.0):.4f})\n\n"
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
    logger.debug(f"Generated Prompt (first 300): {final_prompt[:300]}...")
    return final_prompt
# --- End FIX ---

def get_llm_simplified_vector_response(prompt: str) -> str:
    if not script_openai_client: return "错误：OpenAI client 未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：Chat model 未确定。"
    logger.info(f"Sending prompt to LLM ({ACTIVE_CHAT_MODEL_NAME}). Length: {len(prompt)}")
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME, messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.2,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            ans = response.choices[0].message.content.strip()
            logger.info(f"LLM answer (first 100): {ans[:100]}...")
            return ans
        else: logger.error(f"LLM unexpected response: {response}"); return "抱歉，AI 模型返回格式错误。"
    except RateLimitError as e: logger.error(f"LLM Rate Limit: {e}. Wait 5s..."); time.sleep(5); return f"抱歉，API 速率限制 ({e})"
    except (APIConnectionError, APIError, AuthenticationError) as e: logger.error(f"LLM API Error ({type(e).__name__}): {e}"); return f"抱歉，AI 服务通信出错 ({type(e).__name__})。"
    except Exception as e: logger.error(f"LLM Unexpected Error: {e}", exc_info=True); return f"抱歉，AI 交互未知错误。"

def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    try:
        fp = os.path.join(project_root, file_path); logger.info(f"Loading questions: {fp}")
        with open(fp, 'r', encoding='utf-8') as f: q_list = json.load(f)
        logger.info(f"Loaded {len(q_list)} questions.")
        for i, q in enumerate(q_list): # Validation
            if not all(k in q for k in ['id', 'question', 'ground_truth_ids', 'ground_truth_answer_points']):
                 logger.error(f"Q #{i+1} missing keys."); return []
        return q_list
    except Exception as e: logger.error(f"Load questions error: {e}", exc_info=True); return []


simulation_results: Dict[str, Any] = {
    "simulation_metadata": { "script_name": os.path.basename(__file__), "start_time": datetime.datetime.now().isoformat(), "search_method": "Simplified Vector Search (DB Embeddings + cdist)", "retrieval_top_n": RETRIEVAL_TOP_N, },
    "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting Script ({os.path.basename(__file__)}) ---")
    run_start = time.time()
    if not load_environment_variables(): logger.critical("Env/Client load failed."); sys.exit(1)

    emb_m, emb_meth = get_active_embedding_model_info()
    simulation_results["simulation_metadata"]["embedding_model"] = f"{emb_m} ({emb_meth})"
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME
    logger.info(f"Embedding Model: {emb_m} ({emb_meth})"); logger.info(f"Chat Model: {ACTIVE_CHAT_MODEL_NAME}")

    conn, curs = get_db_connection();
    if not conn: logger.critical("DB connection failed."); sys.exit(1)
    prods_vecs = fetch_products_with_embeddings()
    if not prods_vecs: logger.error("No products with vectors fetched."); close_db_connection(); sys.exit(1)
    simulation_results["simulation_metadata"]["total_products_fetched_with_vector"] = len(prods_vecs)
    logger.info(f"{len(prods_vecs)} product vectors loaded.")

    questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not questions: logger.critical("Failed to load questions."); close_db_connection(); sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions"] = len(questions)

    logger.info("\n--- Running Pipeline ---")
    q_results = []; total_search_t = 0
    for q_data in tqdm(questions, desc="Processing Questions"):
        q_id, q_text, gt_ids, gt_ans = q_data['id'], q_data['question'], q_data['ground_truth_ids'], q_data['ground_truth_answer_points']
        logger.info(f"\nProcessing Q ID: {q_id} - '{q_text}'")
        search_start = time.perf_counter()
        retrieved, _ = find_similar_products_simplified(q_text, prods_vecs, top_n=RETRIEVAL_TOP_N)
        search_end = time.perf_counter(); search_t_ms = (search_end - search_start) * 1000; total_search_t += search_t_ms

        if retrieved is None:
             logger.error(f"Search failed for Q ID {q_id}.")
             res_entry = { "question_id": q_id, "question": q_text, "ground_truth_ids": gt_ids, "ground_truth_answer_points": gt_ans, "status": "Search Error", "retrieved_ids": [], "search_time_ms": search_t_ms, "llm_time_ms": 0, "answer": "ERROR: Search failed." }
             q_results.append(res_entry); continue

        retrieved_ids = [p.get('id') for p in retrieved if p.get('id') is not None]
        logger.info(f"Found {len(retrieved)} products. IDs: {retrieved_ids}. Search Time: {search_t_ms:.2f} ms.")
        prompt = build_simplified_vector_prompt(q_text, retrieved)
        llm_start = time.perf_counter(); answer = get_llm_simplified_vector_response(prompt)
        llm_end = time.perf_counter(); llm_t_ms = (llm_end - llm_start) * 1000
        logger.info(f"LLM time: {llm_t_ms:.2f} ms.")
        res_entry = {
            "question_id": q_id, "question": q_text, "ground_truth_ids": gt_ids, "ground_truth_answer_points": gt_ans,
            "status": "Success", "retrieved_ids": retrieved_ids, "retrieved_details": retrieved,
            "search_time_ms": round(search_t_ms, 2), "llm_time_ms": round(llm_t_ms, 2), "answer": answer
        }
        for detail in res_entry["retrieved_details"]: detail.pop('vector', None); detail.pop('embedding', None)
        q_results.append(res_entry)
        time.sleep(0.2)

    simulation_results["results"] = q_results
    avg_search_t = round(total_search_t / len(questions), 2) if questions else 0
    simulation_results["simulation_metadata"]["average_search_time_ms"] = avg_search_t
    run_end = time.time(); total_run = run_end - run_start
    simulation_results["simulation_metadata"]["end_time"] = datetime.datetime.now().isoformat()
    simulation_results["simulation_metadata"]["total_duration_seconds"] = round(total_run, 2)
    logger.info(f"\n--- Simulation Finished ---"); logger.info(f"Total time: {total_run:.2f}s."); logger.info(f"Avg search time: {avg_search_t:.2f} ms.")

    res_fp = os.path.join(project_root, RESULTS_FILE)
    try:
        logger.info(f"Saving results to {res_fp}...")
        with open(res_fp, 'w', encoding='utf-8') as f: json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info("Results saved.")
    except Exception as e: logger.error(f"Error writing results: {e}", exc_info=True)

    close_db_connection()
    logger.info(f"--- Script Completed ({os.path.basename(__file__)}) ---")