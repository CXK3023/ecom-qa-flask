# simulations/simulate_keyword.py
# Final Version with BM25 Implementation + Corrected Preprocessing

import os
import sys
import json
import re
import time
import logging
from typing import Optional, List, Dict, Any, Tuple 
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import datetime
import decimal

# --- Required Libraries ---
import pandas as pd
from rank_bm25 import BM25Okapi
import spacy
import numpy as np
# import jieba 

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
TEST_QUESTIONS_FILE = 'simulations/test_questions_final.json'
RESULTS_FILE = 'simulations/keyword_simulation_results_bm25_v4.4_corrected.json' # Final results filename
PRODUCT_DATA_CSV_FILE = 'simulations/products_for_eval_v6_cn_v4.4_debug.csv'

ACTIVE_CHAT_MODEL_NAME = "gpt-4.1"
BM25_TOP_K = 5
NAME_WEIGHT_FOR_BM25 = 3
# --- Configuration End ---

# Keep DEBUG level for one more run to verify tokenization
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Global Variables ---
script_openai_client: Optional[OpenAI] = None
NLP_ZH: Optional[spacy.language.Language] = None 

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    global script_openai_client, ACTIVE_CHAT_MODEL_NAME
    # ... (不变) ...
    try:
        env_path = find_dotenv()
        if not env_path: logger.warning(".env file not found.")
        else: load_dotenv(env_path, override=True); logger.info(f".env file loaded from: {env_path}")

        if OpenAI:
            api_key = os.getenv("OPENAI_API_KEY"); api_base = os.getenv("OPENAI_API_BASE")
            if not (api_key and api_base): logger.error("OpenAI API Key or Base URL missing. Cannot initialize client."); return False
            try:
                script_openai_client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("Script's OpenAI client initialized.")
                ACTIVE_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4.1")
                logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")
                return True
            except Exception as client_err: logger.error(f"Failed to initialize script's OpenAI client: {client_err}", exc_info=True); script_openai_client = None; return False
        else: logger.error("OpenAI library not loaded."); return False
    except Exception as e: logger.error(f"Error loading .env or initializing client: {e}", exc_info=True); return False


def load_products_from_csv(csv_file_path: str) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    # ... (不变) ...
    products_list = []
    products_dict = {}
    try:
        full_path = os.path.join(project_root, csv_file_path)
        df = pd.read_csv(full_path)
        logger.info(f"Successfully loaded {len(df)} products from {full_path}")
        required_cols = ['Temp_ID', 'Name', 'Description']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' in CSV {full_path}.")
                return [], {}
        for index, row in df.iterrows():
            product_data = {
                'id': row['Temp_ID'],
                'name': str(row.get('Name', '')),
                'description': str(row.get('Description', '')),
                'price': row.get('Price'),
                'stock': row.get('Stock')
            }
            products_list.append(product_data)
            products_dict[product_data['id']] = product_data
        logger.info(f"Processed {len(products_list)} products into list and dict for BM25.")
        return products_list, products_dict
    except FileNotFoundError: logger.error(f"Product CSV not found: {full_path}"); return [], {}
    except pd.errors.EmptyDataError: logger.error(f"Product CSV is empty: {full_path}"); return [], {}
    except Exception as e: logger.error(f"Error loading products from CSV {full_path}: {e}", exc_info=True); return [], {}


# <<< MODIFIED preprocess_text Function >>>
def preprocess_text(text: str, nlp_model: Optional[spacy.language.Language]) -> List[str]:
    """
    (修正版) 使用 SpaCy 对中文文本进行简化预处理：
    - 分词
    - 转换为小写
    - 去除标点符号
    - 去除空格类字符
    - !! 暂时保留停用词，使用 token.text 而非 lemma_ !!
    返回处理后的 token 列表。
    """
    if not text or not nlp_model:
        if not text: logger.debug("PREPROCESS_INPUT: Empty text string received.")
        if not nlp_model: logger.debug("PREPROCESS_INPUT: SpaCy model not available.")
        return []
    
    doc = nlp_model(str(text))
    processed_tokens = []
    kept_tokens_detail = [] # For richer debug logging

    for token in doc:
        # 简化版标准: 只要不是标点和空格，就保留它的文本形式 (转小写)
        if not token.is_punct and not token.is_space:
            token_text_lower = token.text.lower() # 使用 token.text.lower()
            if token_text_lower: # 确保小写后也不是空字符串
                processed_tokens.append(token_text_lower)
                # 记录更详细的调试信息
                kept_tokens_detail.append(f"{token.text}({token.pos_},{'Stop' if token.is_stop else 'NonStop'})->{token_text_lower}")

    # 更新调试日志
    if text and not text.isspace() and len(text.strip()) > 1:
        logger.debug(f"PREPROCESS_INPUT: '{text[:100].strip().replace(chr(10), ' ')}...'")
        logger.debug(f"PREPROCESS_KEPT_TOKENS (Simpler): {kept_tokens_detail[:30]}") 
        logger.debug(f"PREPROCESS_OUTPUT_TOKENS (Simpler): {processed_tokens[:20]}")
    elif text and (text.isspace() or len(text.strip()) <=1 ):
         logger.debug(f"PREPROCESS_INPUT: (Whitespace or very short text) '{text.replace(chr(10), ' ')}' -> OUTPUT_TOKENS: {processed_tokens[:20]}")

    return processed_tokens

def search_with_bm25(
    query_text: str,
    bm25_model_obj: Optional[BM25Okapi],
    nlp_model: Optional[spacy.language.Language],
    corpus_product_ids: List[Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    # ... (日志部分保持不变) ...
    if not query_text or not bm25_model_obj or not nlp_model or not corpus_product_ids:
        logger.warning("BM25 search called with missing model, NLP, query, or corpus IDs.")
        return []

    logger.info(f"BM25_SEARCH_QUERY_ORIGINAL: '{query_text}'") 
    query_tokens = preprocess_text(query_text, nlp_model) 

    if not query_tokens:
        logger.warning(f"BM25_SEARCH_QUERY_TOKENIZED_EMPTY: Query '{query_text[:50]}...' resulted in no tokens.")
        return []
    logger.info(f"BM25_SEARCH_QUERY_TOKENIZED (Simpler): {query_tokens}") # Indicate simpler preprocessing

    try:
        document_scores = bm25_model_obj.get_scores(query_tokens)
        if document_scores is not None and len(document_scores) > 0:
            logger.debug(f"BM25_DOCUMENT_SCORES (Top 10 raw scores for query '{query_tokens}'): {np.round(document_scores[:10], 2)}")
            logger.debug(f"BM25_DOCUMENT_SCORES_MAX_MIN_AVG: Max={np.max(document_scores):.2f}, Min={np.min(document_scores):.2f}, Avg={np.mean(document_scores):.2f}, Sum={np.sum(document_scores):.2f}")
        else:
            logger.debug(f"BM25_DOCUMENT_SCORES: No scores returned or empty array for query '{query_tokens}'.")
    except Exception as e:
        logger.error(f"Error getting scores from BM25 model: {e}", exc_info=True)
        return []

    num_docs_in_corpus = len(corpus_product_ids)
    actual_top_k = min(top_k, num_docs_in_corpus)
    
    top_n_indices_before_filter = np.argsort(document_scores)[::-1][:actual_top_k]

    scores_before_filter = np.round([document_scores[i] for i in top_n_indices_before_filter], 2)
    ids_before_filter = [corpus_product_ids[i] for i in top_n_indices_before_filter]
    logger.debug(f"BM25_TOP_N_BEFORE_FILTER (Top {actual_top_k}): Scores={scores_before_filter}, IDs={ids_before_filter}")
    
    results = []
    for idx_in_corpus in top_n_indices_before_filter:
        product_id = corpus_product_ids[idx_in_corpus]
        score = document_scores[idx_in_corpus]
        # <<< MODIFIED: Restore score filter >>>
        if score > 0.0: 
            results.append({'id': product_id, 'score': float(score)})
    
    logger.info(f"BM25_SEARCH_RESULTS_COUNT: For query '{query_text[:50]}...', found {len(results)} products (top_k={top_k}, after filtering score > 0.0).") # Updated log message
    if results:
         logger.debug(f"BM25_SEARCH_RESULTS_DETAILS: {[{'id': r['id'], 'score': round(r['score'], 2)} for r in results]}") 
    return results


def build_keyword_prompt(
    question: str, 
    retrieved_items: List[Dict[str, Any]], 
    all_products_data_map: Dict[Any, Dict[str, Any]]
) -> str:
    # ... (不变) ...
    context = "--- 可能相关的商品信息 (基于 BM25 算法匹配) ---\n"
    if not retrieved_items:
        context += "(未找到与问题相关的商品信息)\n"
    else:
        for i, item in enumerate(retrieved_items):
            product_id = item.get('id')
            bm25_score = item.get('score', 0.0)
            product_details = all_products_data_map.get(product_id)
            if product_details:
                context += f"商品 {i+1} (ID: {product_details.get('id', 'N/A')}):\n"
                context += f"  名称: {product_details.get('name', 'N/A')}\n"
                price = product_details.get('price')
                if isinstance(price, (int, float, decimal.Decimal)):
                    try: price_str = f"{price:.2f}"
                    except TypeError: price_str = str(price)
                else: price_str = str(price) if price is not None else "N/A"
                context += f"  价格: {price_str}\n"
                context += f"  库存: {product_details.get('stock', 'N/A')}\n"
                context += f"  (BM25 相关性分数: {bm25_score:.2f})\n\n"
            else:
                logger.warning(f"Could not find product details for ID {product_id} in products map.")
                context += f"商品 {i+1} (ID: {product_id}, BM25 Score: {bm25_score:.2f}): 未找到详细信息。\n\n"
    context += "--- 可能相关的商品信息结束 ---\n\n"
    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面通过“BM25 算法匹配”找到的“可能相关的商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，要求简洁、准确、相关。\n"
        "如果提供的信息包含了用户提问的商品，请优先基于这些信息回答。\n"
        "如果提供的信息不足以回答，或者信息与问题无关，请直接说明“根据我找到的相关信息，无法直接回答您的问题。”或类似措辞。\n"
        "在回答中可以参考BM25相关性分数，但主要依据商品本身的属性信息。\n"
        "禁止编造信息中不存在的内容或价格。"
    )
    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    return final_prompt


def get_llm_keyword_response(prompt: str) -> str:
    # ... (不变) ...
    if not script_openai_client: return "错误：脚本的 OpenAI 客户端未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：无法确定活动的对话模型。"
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.2,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else: 
            logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) returned unexpected response format: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as rle: logger.error(f"LLM Rate Limit: {rle}"); time.sleep(5); return f"API 速率限制，请稍后重试: {rle}"
    except (APIConnectionError, APIError, AuthenticationError) as apie:
         logger.error(f"LLM API Error ({type(apie).__name__}): {apie}")
         return f"AI 模型服务通信出错 ({type(apie).__name__})。"
    except Exception as e: logger.error(f"LLM Unexpected Error: {e}", exc_info=True); return "AI 模型交互时发生未知错误。"


def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    # ... (不变) ...
    try:
        full_path = os.path.join(project_root, file_path)
        with open(full_path, 'r', encoding='utf-8') as f: questions_list = json.load(f)
        logger.info(f"Loaded {len(questions_list)} questions from {full_path}")
        required_keys = ['question_id', 'question', 'ground_truth_ids', 'distractor_ids', 'ground_truth_answer_points']
        valid_questions = []
        for i, q_data in enumerate(questions_list):
             if isinstance(q_data, dict) and all(key in q_data for key in required_keys):
                 if not all(isinstance(q_data[k], list) for k in ['ground_truth_ids', 'distractor_ids', 'ground_truth_answer_points']):
                     logger.warning(f"Q ID {q_data.get('question_id','N/A')} has non-list field(s) among 'ground_truth_ids', 'distractor_ids', 'ground_truth_answer_points'. Skipping.")
                     continue
                 valid_questions.append(q_data)
             else: logger.error(f"Question #{i+1} in {full_path} missing required keys or is not a dict. Skipping.")
        logger.info(f"Loaded {len(valid_questions)} valid questions after format check.")
        return valid_questions
    except Exception as e: logger.error(f"Error loading test questions {full_path}: {e}", exc_info=True); return []


simulation_results: Dict[str, Any] = {
    "simulation_metadata": {
        "script_name": os.path.basename(__file__),
        "script_version": "4.4-bm25_corrected_preprocessing", # Updated version name
        "start_time": datetime.datetime.now().isoformat(),
        "search_method": "BM25 Okapi (Name Weighted, Simpler Preprocessing)", # Updated description
        "bm25_top_k": BM25_TOP_K,
        "bm25_name_weight": NAME_WEIGHT_FOR_BM25,
        "test_questions_file": TEST_QUESTIONS_FILE,
        "product_data_source": PRODUCT_DATA_CSV_FILE,
    }, "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting BM25 Keyword Simulation Script ({os.path.basename(__file__)} - Corrected Preprocessing) ---")
    run_start_time = time.time()

    try:
        NLP_ZH = spacy.load('zh_core_web_sm')
        logger.info("Successfully loaded SpaCy model 'zh_core_web_sm'.")
    except OSError:
        logger.error("SpaCy model 'zh_core_web_sm' not found. Download: python -m spacy download zh_core_web_sm")
        NLP_ZH = None 
    except ImportError:
        logger.error("SpaCy library (spacy) import might have failed.")
        NLP_ZH = None
    simulation_results["simulation_metadata"]["spacy_model_loaded"] = (NLP_ZH is not None)

    if not load_environment_variables():
        logger.critical("Env vars or OpenAI client init failed. Exiting."); sys.exit(1)
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME

    all_products_list, products_by_id_map = load_products_from_csv(PRODUCT_DATA_CSV_FILE)
    if not all_products_list:
        logger.critical(f"No products from CSV: {PRODUCT_DATA_CSV_FILE}. Exiting."); sys.exit(1)
    simulation_results["simulation_metadata"]["total_products_loaded"] = len(all_products_list)

    test_questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not test_questions:
        logger.critical(f"No test questions from {TEST_QUESTIONS_FILE}. Exiting."); sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions_processed"] = len(test_questions)
    
    tokenized_corpus = []
    product_ids_for_corpus = []
    bm25_model = None

    if NLP_ZH:
        logger.info("Preprocessing products for BM25 corpus with name weighting (NAME_WEIGHT=%s)...", NAME_WEIGHT_FOR_BM25)
        processed_product_count_log = 0 
        for product in tqdm(all_products_list, desc="Building BM25 Corpus"):
            product_id = product.get('id')
            name_tokens = preprocess_text(product.get('name', ''), NLP_ZH) # Using corrected preprocess_text
            desc_tokens = preprocess_text(product.get('description', ''), NLP_ZH) # Using corrected preprocess_text
            weighted_tokens = (name_tokens * NAME_WEIGHT_FOR_BM25) + desc_tokens
            
            if processed_product_count_log < 3: 
                logger.debug(f"CORPUS_BUILD_PRODUCT_ID: {product_id}")
                logger.debug(f"CORPUS_BUILD_NAME_RAW: '{product.get('name', '')[:50]}...'")
                # logger.debug(f"CORPUS_BUILD_NAME_TOKENS (x{NAME_WEIGHT_FOR_BM25}): {name_tokens[:10]}") # Log inside preprocess now
                logger.debug(f"CORPUS_BUILD_DESC_RAW: '{product.get('description', '')[:50]}...'")
                # logger.debug(f"CORPUS_BUILD_DESC_TOKENS: {desc_tokens[:10]}") # Log inside preprocess now
                logger.debug(f"CORPUS_BUILD_WEIGHTED_TOKENS (first 20): {weighted_tokens[:20]}")
                processed_product_count_log += 1
            
            tokenized_corpus.append(weighted_tokens)
            product_ids_for_corpus.append(product_id)
        
        if tokenized_corpus:
            logger.info(f"Created tokenized corpus for {len(tokenized_corpus)} products.")
            try:
                # Check if corpus contains only empty lists
                if all(not doc for doc in tokenized_corpus):
                     logger.error("Entire BM25 corpus is empty after preprocessing! Check preprocess_text function and data.")
                     bm25_model = None
                else:
                    bm25_model = BM25Okapi(tokenized_corpus)
                    logger.info("BM25Okapi model initialized successfully.")
            except Exception as e: 
                logger.error(f"Failed to initialize BM25Okapi: {e}", exc_info=True)
                bm25_model = None # Ensure model is None if init fails
        else: 
            logger.warning("Tokenized corpus is empty after processing all products. BM25 model not initialized.")
    else: 
        logger.warning("SpaCy model not loaded. BM25 indexing skipped.")
    simulation_results["simulation_metadata"]["bm25_model_initialized"] = (bm25_model is not None)

    logger.info("\n--- Running BM25 Search Pipeline for Test Questions (Corrected Preprocessing) ---")
    question_results_list = []
    total_search_time_ms = 0
    total_llm_time_ms = 0

    for question_data in tqdm(test_questions, desc="Processing Questions (BM25 Corrected)"):
        # ... (Loop structure remains the same) ...
        q_id = question_data['question_id']
        q_text = question_data['question']
        gt_ids = question_data['ground_truth_ids']
        distractor_ids = question_data['distractor_ids']
        gt_answer_points = question_data['ground_truth_answer_points']

        logger.info(f"--- Processing QID: {q_id} ---") 

        search_start_time = time.perf_counter()
        retrieved_items_with_scores = []
        if bm25_model and NLP_ZH: # Check both models are ready
            retrieved_items_with_scores = search_with_bm25(
                q_text, bm25_model, NLP_ZH, product_ids_for_corpus, top_k=BM25_TOP_K
            )
        else:
            logger.warning(f"BM25 or NLP model not available for QID {q_id}. Search skipped.")
        search_end_time = time.perf_counter()
        search_time_ms = (search_end_time - search_start_time) * 1000
        total_search_time_ms += search_time_ms

        result_entry = {
            "question_id": q_id, "question": q_text, "ground_truth_ids": gt_ids,
            "distractor_ids": distractor_ids, "ground_truth_answer_points": gt_answer_points,
            "search_time_ms": round(search_time_ms, 2), "llm_time_ms": 0,
            "retrieved_ids": [item['id'] for item in retrieved_items_with_scores], 
            "retrieved_details": retrieved_items_with_scores, # Contains [{'id':..., 'score':...}, ...]
            "answer": "", "status": "Processing"
        }
        
        final_prompt = build_keyword_prompt(q_text, retrieved_items_with_scores, products_by_id_map) # Pass map here
        llm_start_time = time.perf_counter()
        llm_answer = get_llm_keyword_response(final_prompt)
        llm_end_time = time.perf_counter()
        llm_time_ms = (llm_end_time - llm_start_time) * 1000
        total_llm_time_ms += llm_time_ms

        result_entry["llm_time_ms"] = round(llm_time_ms, 2)
        result_entry["answer"] = llm_answer
        result_entry["status"] = "Success" if llm_answer and not llm_answer.startswith("错误：") and not llm_answer.startswith("API") else "LLM Error"
        question_results_list.append(result_entry)
        logger.info(f"--- Finished QID: {q_id}, Retrieved {len(result_entry['retrieved_ids'])} items, LLM Answer (first 50 chars): {llm_answer[:50]}... ---")
        time.sleep(0.1) 

    # ... (Final summary and saving results remain the same) ...
    simulation_results["results"] = question_results_list
    num_questions = len(test_questions) if test_questions else 1 
    avg_search_time = round(total_search_time_ms / num_questions, 2) if num_questions > 0 else 0
    avg_llm_time = round(total_llm_time_ms / num_questions, 2) if num_questions > 0 else 0
    simulation_results["simulation_metadata"]["average_search_time_ms"] = avg_search_time
    simulation_results["simulation_metadata"]["average_llm_time_ms"] = avg_llm_time
    run_end_time = time.time()
    total_run_duration = run_end_time - run_start_time
    simulation_results["simulation_metadata"]["end_time"] = datetime.datetime.now().isoformat()
    simulation_results["simulation_metadata"]["total_duration_seconds"] = round(total_run_duration, 2)

    logger.info(f"\n--- BM25 Simulation (Corrected Preprocessing) Finished ---")
    logger.info(f"Total exec time: {total_run_duration:.2f}s. Avg Search: {avg_search_time:.2f}ms. Avg LLM: {avg_llm_time:.2f}ms.")
    results_full_path = os.path.join(project_root, RESULTS_FILE)
    try:
        with open(results_full_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info(f"BM25 results (corrected run) successfully saved to {results_full_path}")
    except Exception as e: logger.error(f"Error writing BM25 results to {results_full_path}: {e}", exc_info=True)
    logger.info(f"--- Script Completed ({os.path.basename(__file__)} - Corrected Preprocessing) ---")