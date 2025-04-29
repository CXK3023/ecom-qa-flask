# simulations/simulate_keyword.py
# Version 4.4 Adaptation: Reads from DB, uses new test_questions format, includes distractor_ids

import os
import sys
import json
import re # 保持用于可能的正则扩展，但当前逻辑是简单包含
import time
import logging
from typing import Optional, List, Dict, Any
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

# --- Configuration ---
# <<< MODIFIED: Point to the new test questions file >>>
TEST_QUESTIONS_FILE = 'simulations/test_questions_final.json' # <-- 使用新的文件名
# <<< MODIFIED: Suggest adding a version suffix to the results file >>>
RESULTS_FILE = 'simulations/keyword_simulation_results_v4.4.json' # <-- 添加版本后缀
# --- Model Configuration ---
ACTIVE_CHAT_MODEL_NAME = "gpt-4.1" # Default, overridden by env var
# --- Keyword Search Configuration ---
KEYWORD_MATCH_LIMIT = 5 # Limit the number of matched items to include in prompt
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
    """ Custom JSON encoder to handle Decimal types """
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        # Keyword script doesn't handle bytes or ndarrays typically
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    """Load .env variables and initialize the script's OpenAI Client."""
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
    """Establish and return a database connection."""
    global db_connection, db_cursor
    if db_connection and db_connection.is_connected(): return db_connection, db_cursor
    db_url = os.getenv('DATABASE_URL')
    if not db_url: logger.critical("DATABASE_URL not found."); return None, None
    try:
        from urllib.parse import urlparse; parsed_url = urlparse(db_url)
        db_user = parsed_url.username; db_password = parsed_url.password; db_host = parsed_url.hostname
        db_port = parsed_url.port or 3306; db_name = parsed_url.path.lstrip('/')
        logger.info(f"Connecting to DB: {db_host}:{db_port}/{db_name} as {db_user}")
        db_connection = mysql.connector.connect(host=db_host, port=db_port, user=db_user, password=db_password, database=db_name)
        db_cursor = db_connection.cursor(dictionary=True)
        logger.info("Database connection established.")
        return db_connection, db_cursor
    except mysql.connector.Error as err: logger.error(f"Database connection failed: {err}", exc_info=True); db_connection = None; db_cursor = None; return None, None
    except Exception as e: logger.error(f"Unexpected error during DB connection: {e}", exc_info=True); db_connection = None; db_cursor = None; return None, None

def close_db_connection():
    """Close the database connection and cursor."""
    global db_connection, db_cursor
    if db_cursor: db_cursor.close(); logger.info("DB cursor closed.")
    if db_connection and db_connection.is_connected(): db_connection.close(); logger.info("DB connection closed.")
    db_cursor = None; db_connection = None

def fetch_products_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch product data (id, name, description, price, stock) from the database for keyword search."""
    conn, cursor = get_db_connection()
    if not conn or not cursor: return []
    products = []
    try:
        # Fetch necessary fields for keyword matching and context building
        query = "SELECT id, name, description, price, stock FROM product WHERE status = 'active'"
        if limit: query += f" LIMIT {limit}"
        cursor.execute(query)
        products = cursor.fetchall()
        logger.info(f"Fetched {len(products)} active products from database for keyword search.")
    except mysql.connector.Error as err: logger.error(f"Error fetching products for keyword search: {err}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error fetching products for keyword search: {e}", exc_info=True)

    # Ensure name and description are strings and not None for matching
    for p in products:
        if p.get('name') is None: p['name'] = ''
        if p.get('description') is None: p['description'] = ''
        # Price/stock handled during prompt building
    return products

# --- Keyword Search Logic ---
def search_by_keyword(query: str, products: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a simple case-insensitive keyword search on product name and description.
    Matches if the query string is found as a substring. Limits results.
    """
    if not query or not products: return [] # Handle empty query or products
    matched_products = []
    query_lower = query.lower().strip() # Normalize query
    if not query_lower: return [] # Handle empty query after stripping

    logger.info(f"Searching for keyword phrase (case-insensitive substring): '{query_lower}'")

    for product in products:
        # Combine name and description for searching
        name_lower = str(product.get('name', '')).lower()
        desc_lower = str(product.get('description', '')).lower()
        search_text = name_lower + " " + desc_lower

        # Simple substring check
        if query_lower in search_text:
            # Add a copy to avoid modifying the original list directly if needed elsewhere
            matched_products.append(product.copy())
            if len(matched_products) >= limit: # Check limit
                 logger.info(f"Keyword match limit ({limit}) reached.")
                 break # Stop searching once limit is hit

    logger.info(f"Keyword search found {len(matched_products)} matching products (up to limit {limit}).")
    return matched_products
# --- End Keyword Search Logic ---

def build_keyword_prompt(question: str, matched_docs: List[Dict[str, Any]]) -> str:
    """Build the prompt including matched context for the LLM."""
    context = "--- 可能相关的商品信息 (基于关键词匹配) ---\n"
    if not matched_docs:
        context += "(未找到关键词匹配的商品信息)\n"
    else:
        for i, doc in enumerate(matched_docs):
            context += f"商品 {i+1} (ID: {doc.get('id', 'N/A')}):\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"
            # Optional: Include description snippet if useful
            # desc_snippet = str(doc.get('description', ''))[:100] + ('...' if len(str(doc.get('description', ''))) > 100 else '')
            # context += f"  描述片段: {desc_snippet}\n"

            # Format price safely
            price = doc.get('price')
            if isinstance(price, (int, float, decimal.Decimal)):
                try: price_str = f"{price:.2f}"
                except TypeError: price_str = str(price)
            else: price_str = str(price) if price is not None else "N/A"
            context += f"  价格: {price_str}\n"

            context += f"  库存: {doc.get('stock', 'N/A')}\n\n" # Add extra newline for spacing
    context += "--- 可能相关的商品信息结束 ---\n\n"

    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面通过“关键词匹配”找到的“可能相关的商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，要求简洁、准确、相关。\n"
        "如果信息看起来相关，请基于它们回答。如果信息明显不相关或不足以回答，请说明无法根据找到的信息回答。\n"
        "注意：这些信息是基于简单的文本匹配找到的，可能不是语义上最相关的。\n"
        "禁止编造信息中不存在的内容或价格。"
    )

    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    logger.debug(f"Generated Keyword Prompt (first 300 chars): {final_prompt[:300]}...")
    return final_prompt

# --- Reuse LLM call function (can be renamed if preferred) ---
def get_llm_keyword_response(prompt: str) -> str:
    """Get the response from the LLM based on the keyword prompt."""
    if not script_openai_client: return "错误：脚本的 OpenAI 客户端未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：无法确定活动的对话模型。"
    logger.info(f"Sending prompt to LLM ({ACTIVE_CHAT_MODEL_NAME}). Length: {len(prompt)}")
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3, # Keyword matches might be less precise, slightly higher temp?
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) generated keyword answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) returned an unexpected keyword response format: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as rle:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Rate Limit Error for keyword: {rle}. Waiting 5s...")
        time.sleep(5); return f"抱歉，已达到 API 速率限制，请稍后重试。 ({rle})"
    except (APIConnectionError, APIError, AuthenticationError) as apie:
         logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) API Error for keyword ({type(apie).__name__}): {apie}")
         return f"抱歉，与 AI 模型服务通信时出错 ({type(apie).__name__})。"
    except Exception as e:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Unexpected Error during keyword API call: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误。"
# --- End LLM call function ---

# <<< MODIFIED: Reuse the updated load_test_questions function >>>
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
        "search_method": "Keyword Substring Match",
        "match_limit": KEYWORD_MATCH_LIMIT,
        "test_questions_file": TEST_QUESTIONS_FILE, # Record questions file
    },
    "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting Keyword Search Simulation Script ({os.path.basename(__file__)}) ---")
    run_start_time = time.time()

    # 1. Load Environment & Client
    if not load_environment_variables():
        logger.critical("Failed to load environment or initialize OpenAI client. Exiting.")
        sys.exit(1)

    # Add model info to metadata
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME
    logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")

    # 2. Connect to DB & Fetch Products
    conn, cursor = get_db_connection()
    if not conn: logger.critical("Database connection failed. Exiting."); sys.exit(1)

    products = fetch_products_from_db() # Fetch all active products for matching
    if not products:
        logger.warning("No active products fetched from database. Keyword search will yield no results.")
        # Allow running even with 0 products
    simulation_results["simulation_metadata"]["total_products_fetched_from_db"] = len(products)

    # 3. Load Test Questions
    test_questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not test_questions:
        logger.critical(f"Failed to load valid test questions from {TEST_QUESTIONS_FILE}. Exiting.")
        close_db_connection()
        sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions_processed"] = len(test_questions)


    # 4. Run Keyword Search Pipeline for each question
    logger.info("\n--- Running Keyword Search Pipeline for Test Questions ---")
    question_results_list = []
    total_search_time_ms = 0
    total_llm_time_ms = 0

    for question_data in tqdm(test_questions, desc="Processing Questions (Keyword Search)"):
        # <<< MODIFIED: Unpack new fields >>>
        q_id = question_data['question_id']
        q_text = question_data['question']
        gt_ids = question_data['ground_truth_ids']
        distractor_ids = question_data['distractor_ids'] # Extract distractors
        gt_answer_points = question_data['ground_truth_answer_points']
        # <<< END MODIFIED >>>

        logger.info(f"\nProcessing Question ID: {q_id} - '{q_text[:60]}...'")

        # --- Keyword Search Phase ---
        search_start_time = time.perf_counter()
        # Pass the full product list for searching
        matched_products = search_by_keyword(
            q_text, products, limit=KEYWORD_MATCH_LIMIT
        )
        search_end_time = time.perf_counter()
        search_time_ms = (search_end_time - search_start_time) * 1000
        total_search_time_ms += search_time_ms

        # Prepare result entry structure
        result_entry = {
            "question_id": q_id,
            "question": q_text,
            "ground_truth_ids": gt_ids,
            "distractor_ids": distractor_ids, # <<< MODIFIED: Include distractor_ids >>>
            "ground_truth_answer_points": gt_answer_points,
            "search_time_ms": round(search_time_ms, 2),
            "llm_time_ms": 0,
            "matched_ids": [], # Use 'matched_ids' for keyword results
            "matched_details": [], # Store details of matched products
            "answer": "",
            "status": "Processing"
        }

        # Get matched IDs and details
        matched_product_ids = [p.get('id') for p in matched_products if p.get('id') is not None]
        result_entry["matched_ids"] = matched_product_ids
        result_entry["matched_details"] = matched_products # Store the matched product dicts

        logger.info(f"Found {len(matched_products)} matched products. IDs: {matched_product_ids}. Search Time: {search_time_ms:.2f} ms.")

        # --- LLM Phase ---
        final_prompt = build_keyword_prompt(q_text, matched_products)

        llm_start_time = time.perf_counter()
        llm_answer = get_llm_keyword_response(final_prompt)
        llm_end_time = time.perf_counter()
        llm_time_ms = (llm_end_time - llm_start_time) * 1000
        total_llm_time_ms += llm_time_ms

        result_entry["llm_time_ms"] = round(llm_time_ms, 2)
        result_entry["answer"] = llm_answer
        result_entry["status"] = "Success"

        logger.info(f"LLM generation time: {llm_time_ms:.2f} ms.")

        question_results_list.append(result_entry)

        # Optional delay
        time.sleep(0.2) # Keyword search is local, shorter delay ok

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

    logger.info(f"\n--- Keyword Simulation Finished ---")
    logger.info(f"Total execution time: {total_run_duration:.2f} seconds.")
    logger.info(f"Average Search Time: {avg_search_time:.2f} ms per question.")
    logger.info(f"Average LLM Time: {avg_llm_time:.2f} ms per question.")

    # Save results
    results_full_path = os.path.join(project_root, RESULTS_FILE)
    try:
        logger.info(f"Saving keyword results to {results_full_path}...")
        with open(results_full_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info(f"Keyword results successfully saved to {results_full_path}")
    except Exception as e:
        logger.error(f"Error writing keyword results to {results_full_path}: {e}", exc_info=True)

    # 6. Close Database Connection
    close_db_connection()

    logger.info(f"--- Keyword Simulation Script Completed ({os.path.basename(__file__)}) ---")