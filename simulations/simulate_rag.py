# simulations/simulate_rag.py
# Version 3.2: Uses Persistent ChromaDB to avoid regeneration if index exists

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
    # Import specific exception if needed, or use general Exception
    # from chromadb.errors import CollectionNotFoundError
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
    OpenAI = None
    APIError = None
    RateLimitError = None
    APIConnectionError = None
    AuthenticationError = None

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
    get_active_embedding_model_info = lambda: (os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "doubao-embedding-large-text"), "remote_api")
    def generate_embedding(text: str) -> Optional[bytes]:
         logger.warning("Using fallback generate_embedding. Ensure script's OpenAI client is initialized.")
         if not script_openai_client:
             logger.error("Fallback generate_embedding failed: script_openai_client not initialized.")
             return None
         try:
             model_name, _ = get_active_embedding_model_info()
             response = script_openai_client.embeddings.create(model=model_name, input=[text])
             if response.data and response.data[0].embedding:
                 vector = np.array(response.data[0].embedding)
                 import pickle
                 return pickle.dumps(vector)
             return None
         except Exception as emb_err:
              logger.error(f"Fallback embedding generation failed: {emb_err}")
              return None
    def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
         import pickle
         if not serialized_embedding: return None
         try:
             vector = pickle.loads(serialized_embedding)
             return vector if isinstance(vector, np.ndarray) else None
         except Exception as des_err:
             logger.error(f"Fallback deserializing failed: {des_err}")
             return None

# --- Configuration ---
TEST_QUESTIONS_FILE = 'simulations/test_questions.json'
RESULTS_FILE = 'simulations/rag_simulation_results_persistent.json' # Different results file
# --- Model Configuration ---
ACTIVE_CHAT_MODEL_NAME = "gpt-4.1" # Default, overridden by env var
# --- ChromaDB Configuration ---
CHROMA_DB_PATH = "./chroma_data_rag" # <-- Path for persistent storage
CHROMA_COLLECTION_NAME = "products_rag_persistent_v3_2" # Use a distinct name
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
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def load_environment_variables():
    global script_openai_client, ACTIVE_CHAT_MODEL_NAME
    try:
        env_path = find_dotenv()
        if not env_path:
            logger.warning(".env file not found.")
        else:
            load_dotenv(env_path, override=True)
            logger.info(f".env file loaded from: {env_path}")

        if OpenAI:
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE")
            if not (api_key and api_base):
                logger.error("OpenAI API Key or Base URL missing in .env file. Cannot initialize client.")
                return False
            try:
                script_openai_client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("Script's OpenAI client initialized successfully.")
                ACTIVE_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4.1")
                logger.info(f"Using Chat Model (from env or default): {ACTIVE_CHAT_MODEL_NAME}")
                return True
            except Exception as client_err:
                logger.error(f"Failed to initialize script's OpenAI client: {client_err}", exc_info=True)
                script_openai_client = None
                return False
        else:
            logger.error("OpenAI library not loaded. Cannot initialize client.")
            return False
    except Exception as e:
        logger.error(f"Error loading .env file or initializing client: {e}", exc_info=True)
        return False

def get_db_connection():
    global db_connection, db_cursor
    if db_connection and db_connection.is_connected():
        return db_connection, db_cursor
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.critical("DATABASE_URL not found.")
        return None, None
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(db_url)
        db_user = parsed_url.username
        db_password = parsed_url.password
        db_host = parsed_url.hostname
        db_port = parsed_url.port
        db_name = parsed_url.path.lstrip('/')
        logger.info(f"Connecting to DB: {db_host}:{db_port}/{db_name} as {db_user}")
        db_connection = mysql.connector.connect(
            host=db_host, port=db_port, user=db_user, password=db_password, database=db_name
        )
        db_cursor = db_connection.cursor(dictionary=True)
        logger.info("Database connection established.")
        return db_connection, db_cursor
    except Exception as e:
        logger.error(f"Database connection failed: {e}", exc_info=True)
        db_connection = None
        db_cursor = None
        return None, None

def close_db_connection():
    global db_connection, db_cursor
    if db_cursor: db_cursor.close(); logger.info("DB cursor closed.")
    if db_connection and db_connection.is_connected(): db_connection.close(); logger.info("DB connection closed.")
    db_cursor = None; db_connection = None

def fetch_products_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    conn, cursor = get_db_connection()
    if not conn or not cursor: return []
    products = []
    try:
        query = "SELECT id, name, description, price, stock FROM product WHERE status = 'active'"
        if limit: query += f" LIMIT {limit}"
        cursor.execute(query)
        products = cursor.fetchall()
        logger.info(f"Fetched {len(products)} active products from database.")
    except mysql.connector.Error as err:
        logger.error(f"Error fetching products: {err}", exc_info=True)
    for p in products:
        if p.get('description') is None: p['description'] = ''
    return products

def generate_single_embedding_wrapper(text: str) -> Optional[List[float]]:
    if not text: return None
    serialized_vector = generate_embedding(text)
    if serialized_vector:
        vector = deserialize_embedding(serialized_vector)
        if vector is not None:
            return vector.tolist()
    return None

# --- Modified: Use PersistentClient and load/create logic ---
def setup_chroma_vector_store(products: List[Dict[str, Any]]) -> Optional[chromadb.Collection]:
    """
    Setup ChromaDB persistent store. Loads existing collection if found,
    otherwise creates it and generates embeddings.
    """
    if not chromadb:
        logger.error("chromadb library is not available.")
        return None

    logger.info(f"Setting up ChromaDB persistent store at: {CHROMA_DB_PATH}")
    try:
        # --- Use PersistentClient ---
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # --- Try to get existing collection ---
        try:
            collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            logger.info(f"Loaded existing ChromaDB collection: '{CHROMA_COLLECTION_NAME}'")

            # --- Basic Sanity Check ---
            existing_count = collection.count()
            db_count = len(products)
            logger.info(f"Existing collection has {existing_count} items. Current DB query returned {db_count} products.")
            if abs(existing_count - db_count) > db_count * 0.1: # Allow 10% difference? Adjust threshold
                 logger.warning(f"Significant difference between items in existing ChromaDB ({existing_count}) and current DB product count ({db_count}). Index might be stale.")
            # NOTE: No automatic update/rebuild implemented here.
            # If you need to rebuild, manually delete the CHROMA_DB_PATH directory.
            # --- End Sanity Check ---

            # Store metadata about loading
            global simulation_results
            if 'metadata' not in simulation_results: simulation_results['metadata'] = {}
            simulation_results['metadata']['indexing_action'] = "Loaded Existing"
            simulation_results['metadata']['items_in_loaded_index'] = existing_count

            return collection

        except Exception as e: # Catches CollectionNotFoundError and potentially others
            logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' not found or error getting it ({type(e).__name__}). Will create a new one.")

            # --- Collection not found, create it ---
            collection = chroma_client.get_or_create_collection(
                CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"} # Specify cosine distance
            )
            logger.info(f"Created new ChromaDB collection: '{CHROMA_COLLECTION_NAME}' with cosine distance.")

            # --- Proceed with embedding generation and adding data ---
            if not products:
                logger.error("No products fetched from DB, cannot populate new collection.")
                return None

            ids_to_add, embeddings_to_add, documents_to_add, metadatas_to_add = [], [], [], []
            failed_embeddings = 0
            start_time = time.time()

            logger.info("Generating embeddings and preparing data for new ChromaDB collection...")
            for product in tqdm(products, desc="Processing Products for ChromaDB"):
                product_id_str = str(product['id'])
                name_str = str(product.get('name', ''))
                desc_str = str(product.get('description', ''))
                text_to_embed = f"商品名称: {name_str}\n商品描述: {desc_str}"
                embedding = generate_single_embedding_wrapper(text_to_embed)

                if embedding is None or not isinstance(embedding, list):
                    logger.warning(f"Skipping product ID {product_id_str} due to embedding generation failure.")
                    failed_embeddings += 1
                    continue

                ids_to_add.append(product_id_str)
                embeddings_to_add.append(embedding)
                documents_to_add.append(text_to_embed)
                metadata = {
                    "original_id": int(product['id']),
                    "name": name_str,
                    "price": float(product.get('price', 0.0) or 0.0),
                    "stock": int(product.get('stock', 0) or 0)
                }
                metadatas_to_add.append(metadata)

            if not ids_to_add:
                logger.error("No valid data to add to ChromaDB after embedding generation.")
                # Cleanup potentially created empty collection? Optional.
                # try: chroma_client.delete_collection(CHROMA_COLLECTION_NAME) except: pass
                return None

            logger.info(f"Adding {len(ids_to_add)} items to ChromaDB collection (Embeddings failed for {failed_embeddings} items)...")
            # Use upsert=True? No, we are creating new, so add is fine.
            # If we were implementing update logic, upsert would be better.
            batch_size = 5000
            for i in range(0, len(ids_to_add), batch_size):
                collection.add(
                    ids=ids_to_add[i:i+batch_size],
                    embeddings=embeddings_to_add[i:i+batch_size],
                    documents=documents_to_add[i:i+batch_size],
                    metadatas=metadatas_to_add[i:i+batch_size]
                )
            end_time = time.time()
            indexing_time = end_time - start_time
            logger.info(f"Finished adding data. Indexing time: {indexing_time:.2f} seconds.")

            item_count = collection.count()
            logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' now contains {item_count} items.")

            # Store indexing time for results
            # global simulation_results (already declared global)
            if 'metadata' not in simulation_results: simulation_results['metadata'] = {}
            simulation_results['metadata']['indexing_action'] = "Created New"
            simulation_results['metadata']['indexing_time_seconds'] = indexing_time
            simulation_results['metadata']['failed_embeddings_indexing'] = failed_embeddings
            simulation_results['metadata']['items_indexed'] = item_count

            return collection

    except Exception as e:
        logger.error(f"Fatal error during ChromaDB setup: {e}", exc_info=True)
        return None
# --- End of modified setup_chroma_vector_store ---

def retrieve_relevant_docs(
    question_text: str, collection: chromadb.Collection, top_n: int = 3
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[float]]]:
    if not collection: logger.error("ChromaDB collection unavailable."); return None, None
    logger.info(f"Retrieving documents for question: '{question_text}'")
    question_embedding = generate_single_embedding_wrapper(question_text)
    if question_embedding is None:
        logger.error("Failed to generate question embedding.")
        return None, None
    try:
        results: QueryResult = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_n,
            include=['metadatas', 'distances']
        )
        logger.debug(f"ChromaDB query raw results: {results}")
        retrieved_docs = []
        if results and results.get('ids') and results['ids'][0]:
            ids, metadatas, distances = results['ids'][0], results.get('metadatas', [[]])[0], results.get('distances', [[]])[0]
            for i in range(len(ids)):
                distance = distances[i] if distances and distances[i] is not None else -1.0
                similarity = 1.0 - distance if distance >= 0 else 0.0
                doc_info = {
                    "chroma_id": ids[i], "metadata": metadatas[i] if metadatas else {},
                    "distance": distance, "similarity": similarity,
                    "id": metadatas[i].get("original_id") if metadatas else None,
                    "name": metadatas[i].get("name") if metadatas else "N/A",
                    "price": metadatas[i].get("price") if metadatas else "N/A",
                    "stock": metadatas[i].get("stock") if metadatas else "N/A",
                }
                retrieved_docs.append(doc_info)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            retrieved_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return retrieved_docs, question_embedding
        else:
            logger.info("No relevant documents found by ChromaDB query.")
            return [], question_embedding
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        return None, question_embedding

def build_rag_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = "--- 相关商品信息 ---\n"
    if not retrieved_docs:
        context += "(未找到相关商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1} (ID: {doc.get('id', 'N/A')}):\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"
            context += f"  价格: {doc.get('price', 'N/A'):.2f}\n"
            context += f"  库存: {doc.get('stock', 'N/A')}\n"
            context += f"  (检索相似度: {doc.get('similarity', 0.0):.4f})\n\n"
    context += "--- 相关商品信息结束 ---\n\n"
    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面提供的“相关商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，要求简洁、准确、相关。\n"
        "如果提供的信息包含了用户提问的商品，请优先基于这些信息回答。\n"
        "如果提供的信息不足以回答，或者信息与问题无关，请直接说明“根据我找到的相关信息，无法直接回答您的问题。”或类似措辞。\n"
        "禁止编造信息中不存在的内容或价格。"
    )
    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    logger.debug(f"Generated RAG Prompt (first 300 chars): {final_prompt[:300]}...")
    return final_prompt

def get_llm_rag_response(prompt: str) -> str:
    if not script_openai_client: return "错误：脚本的 OpenAI 客户端未初始化。"
    if not ACTIVE_CHAT_MODEL_NAME: return "错误：无法确定活动的对话模型。"
    logger.info(f"Sending prompt to LLM ({ACTIVE_CHAT_MODEL_NAME}). Length: {len(prompt)}")
    try:
        response = script_openai_client.chat.completions.create(
            model=ACTIVE_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.2,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) generated answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) returned an unexpected response: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"
    except RateLimitError as e:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Rate Limit Error: {e}. Waiting 5s...")
        time.sleep(5); return f"抱歉，达到 API 速率限制。 ({e})" # Simplified retry handling
    except (APIConnectionError, APIError, AuthenticationError) as e:
         logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) API Error ({type(e).__name__}): {e}")
         return f"抱歉，与 AI 模型服务通信时出错 ({type(e).__name__})。"
    except Exception as e:
        logger.error(f"LLM ({ACTIVE_CHAT_MODEL_NAME}) Unexpected Error: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误。"

def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    try:
        full_path = os.path.join(project_root, file_path)
        logger.info(f"Attempting to load test questions from: {full_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        logger.info(f"Successfully loaded {len(questions)} questions from {full_path}")
        for i, q in enumerate(questions):
            if not all(k in q for k in ['id', 'question', 'ground_truth_ids', 'ground_truth_answer_points']):
                 logger.error(f"Question #{i+1} in {full_path} is missing required keys.")
                 return []
        return questions
    except Exception as e:
        logger.error(f"Error loading test questions from {full_path}: {e}", exc_info=True)
        return []

# --- Main Execution ---
simulation_results: Dict[str, Any] = {
    "simulation_metadata": {
        "script_name": os.path.basename(__file__),
        "start_time": datetime.datetime.now().isoformat(),
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "chroma_db_path": CHROMA_DB_PATH, # Record path
        "retrieval_top_n": RETRIEVAL_TOP_N,
    },
    "results": []
}

if __name__ == "__main__":
    logger.info(f"--- Starting RAG Simulation Script (Persistent DB Version {os.path.basename(__file__)} ) ---")
    run_start_time = time.time()

    if not load_environment_variables():
        logger.critical("Failed to load environment or initialize OpenAI client. Exiting.")
        sys.exit(1)

    emb_model, emb_method = get_active_embedding_model_info()
    simulation_results["simulation_metadata"]["embedding_model"] = f"{emb_model} ({emb_method})"
    simulation_results["simulation_metadata"]["chat_model"] = ACTIVE_CHAT_MODEL_NAME
    logger.info(f"Using Embedding Model: {emb_model} ({emb_method})")
    logger.info(f"Using Chat Model: {ACTIVE_CHAT_MODEL_NAME}")

    conn, cursor = get_db_connection()
    if not conn: logger.critical("Database connection failed. Exiting."); sys.exit(1)

    products = fetch_products_from_db()
    if not products:
        logger.error("No active products fetched from database. Cannot proceed.")
        close_db_connection()
        sys.exit(1)
    simulation_results["simulation_metadata"]["total_products_fetched"] = len(products)

    # --- Setup ChromaDB (Loads existing or Creates new) ---
    product_collection = setup_chroma_vector_store(products)
    if not product_collection:
        logger.critical("Failed to setup ChromaDB vector store. Exiting.")
        close_db_connection()
        sys.exit(1)
    # --- Metadata about indexing action is added within setup_chroma_vector_store ---

    test_questions = load_test_questions(TEST_QUESTIONS_FILE)
    if not test_questions:
        logger.critical(f"Failed to load test questions from {TEST_QUESTIONS_FILE}. Exiting.")
        close_db_connection()
        sys.exit(1)
    simulation_results["simulation_metadata"]["total_questions"] = len(test_questions)

    logger.info("\n--- Running RAG Pipeline for Test Questions ---")
    question_results = []
    for question_data in tqdm(test_questions, desc="Processing Questions"):
        question_id = question_data['id']
        question_text = question_data['question']
        gt_ids = question_data['ground_truth_ids']
        gt_answer_points = question_data['ground_truth_answer_points']

        logger.info(f"\nProcessing Question ID: {question_id} - '{question_text}'")

        retrieval_start_time = time.perf_counter()
        retrieved_docs, q_embedding = retrieve_relevant_docs(
            question_text, product_collection, top_n=RETRIEVAL_TOP_N
        )
        retrieval_end_time = time.perf_counter()
        retrieval_time_ms = (retrieval_end_time - retrieval_start_time) * 1000

        if retrieved_docs is None:
             logger.error(f"Retrieval failed for question ID {question_id}.")
             result_entry = { "question_id": question_id, "question": question_text, "ground_truth_ids": gt_ids, "ground_truth_answer_points": gt_answer_points, "status": "Retrieval Error", "retrieved_ids": [], "retrieval_time_ms": retrieval_time_ms, "llm_time_ms": 0, "answer": "ERROR: Retrieval failed." }
             question_results.append(result_entry)
             continue

        retrieved_product_ids = [doc.get('id') for doc in retrieved_docs if doc.get('id') is not None]
        logger.info(f"Retrieved {len(retrieved_docs)} docs. IDs: {retrieved_product_ids}. Time: {retrieval_time_ms:.2f} ms.")

        final_prompt = build_rag_prompt(question_text, retrieved_docs)
        llm_start_time = time.perf_counter()
        llm_answer = get_llm_rag_response(final_prompt)
        llm_end_time = time.perf_counter()
        llm_time_ms = (llm_end_time - llm_start_time) * 1000
        logger.info(f"LLM generation time: {llm_time_ms:.2f} ms.")

        result_entry = {
            "question_id": question_id, "question": question_text, "ground_truth_ids": gt_ids,
            "ground_truth_answer_points": gt_answer_points, "status": "Success",
            "retrieved_ids": retrieved_product_ids, "retrieved_details": retrieved_docs,
            "retrieval_time_ms": round(retrieval_time_ms, 2), "llm_time_ms": round(llm_time_ms, 2),
            "answer": llm_answer
        }
        question_results.append(result_entry)
        time.sleep(0.5)

    simulation_results["results"] = question_results
    run_end_time = time.time()
    total_run_time = run_end_time - run_start_time
    simulation_results["simulation_metadata"]["end_time"] = datetime.datetime.now().isoformat()
    simulation_results["simulation_metadata"]["total_duration_seconds"] = round(total_run_time, 2)
    logger.info(f"\n--- Simulation Finished ---")
    logger.info(f"Total execution time: {total_run_time:.2f} seconds.")

    results_full_path = os.path.join(project_root, RESULTS_FILE)
    try:
        logger.info(f"Saving results to {results_full_path}...")
        with open(results_full_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_results, f, ensure_ascii=False, indent=4, cls=DecimalEncoder)
        logger.info(f"Results successfully saved.")
    except Exception as e:
        logger.error(f"Error writing results to {results_full_path}: {e}", exc_info=True)

    close_db_connection()
    logger.info("--- RAG Simulation Script Completed ---")