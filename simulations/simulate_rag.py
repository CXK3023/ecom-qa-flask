# simulations/simulate_rag.py

import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm
import time # 引入 time 模块

# --- ChromaDB 导入 ---
try:
    import chromadb
    from chromadb.api.types import QueryResult
except ImportError:
    logging.error("chromadb library not found. Please install it: pip install chromadb")
    chromadb = None
    QueryResult = None # type: ignore
# --- ChromaDB 导入结束 ---


# --- 配置 ---
CSV_FILE_PATH = 'products_export.csv'
# --- 模型配置 ---
ACTIVE_EMBEDDING_MODEL_NAME = 'doubao-embedding-large-text'
ACTIVE_EMBEDDING_INVOCATION_METHOD = 'remote_api'
# ! 修改为您 Day 12 设置的活动对话模型名称
ACTIVE_CHAT_MODEL_NAME = 'gpt-4.1'
# --- ChromaDB 配置 ---
CHROMA_COLLECTION_NAME = "products_rag_sim"
# --- 检索与 RAG 配置 ---
RETRIEVAL_TOP_N = 3
# --- 配置结束 ---

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- 日志设置结束 ---

# --- 库导入和客户端初始化 ---
SentenceTransformer = None
OpenAI = None
APIError = None
openai_client = None # 同时用于 Embedding 和 Chat
local_model_cache = {}

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence-transformers library not found.")

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError # 添加更多错误类型
    # OpenAI client 在 load_environment_variables 中初始化
except ImportError:
    logger.warning("openai library not found.")
# --- 库导入和客户端初始化结束 ---


def load_environment_variables():
    """加载 .env 文件中的环境变量并初始化 OpenAI Client"""
    global openai_client
    try:
        env_path = find_dotenv()
        if not env_path:
             logger.warning(".env file not found.")
             return False # 返回失败状态
        load_dotenv(env_path, override=True)
        logger.info(f".env file loaded successfully from: {env_path}")

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")

        if not (api_key and api_base):
            logger.error("OpenAI API Key or Base URL missing in .env file.")
            return False

        if not OpenAI:
             logger.error("OpenAI library not loaded, cannot initialize client.")
             return False

        # 初始化 Client (用于 Embedding 和 Chat)
        try:
             openai_client = OpenAI(api_key=api_key, base_url=api_base)
             logger.info("OpenAI client initialized successfully.")
             # 可以在这里尝试一个简单的 ping 或模型列表调用来验证连接和 key
             # try:
             #      openai_client.models.list()
             #      logger.info("OpenAI client connection verified.")
             # except Exception as ping_err:
             #      logger.error(f"OpenAI client connection verification failed: {ping_err}")
             #      openai_client = None # 验证失败则设为 None
             #      return False
             return True # 初始化成功
        except Exception as client_err:
              logger.error(f"Failed to initialize OpenAI client: {client_err}", exc_info=True)
              openai_client = None
              return False

    except Exception as e:
        logger.error(f"Error loading .env file or initializing client: {e}", exc_info=True)
        return False

# --- load_product_data, _load_local_embedding_model, generate_embeddings_for_products, setup_chroma_vector_store 函数 (保持不变) ---
def load_product_data(csv_relative_path: str) -> Optional[pd.DataFrame]:
    # ... (代码不变) ...
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

def _load_local_embedding_model(model_name: str) -> Optional[SentenceTransformer]:
    # ... (代码不变) ...
    global local_model_cache
    if not SentenceTransformer:
         logger.error("SentenceTransformer library is required but not installed.")
         return None
    if model_name in local_model_cache:
        logger.debug(f"Using cached local model: {model_name}")
        return local_model_cache[model_name]
    try:
        logger.info(f"Loading local SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name)
        local_model_cache[model_name] = model
        logger.info(f"Successfully loaded and cached local model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading local SentenceTransformer model '{model_name}': {e}", exc_info=True)
        return None

def generate_embeddings_for_products(df: pd.DataFrame, model_name: str, invocation_method: str) -> Optional[List]:
    # ... (代码不变) ...
    embeddings_list: List[Optional[List[float]]] = []
    texts_to_embed = []
    logger.info(f"Preparing text for embedding using model '{model_name}' ({invocation_method})...")
    for index, row in df.iterrows():
        name_str = str(row.get('Name', ''))
        desc_str = str(row.get('Description', ''))
        text = f"商品名称: {name_str}\n商品描述: {desc_str}"
        texts_to_embed.append(text)
    logger.info(f"Starting embedding generation for {len(texts_to_embed)} products...")
    if invocation_method == 'local':
        model = _load_local_embedding_model(model_name)
        if not model: return None
        try:
            raw_embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=32)
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else list(emb) for emb in raw_embeddings]
            logger.info(f"Successfully generated {len(embeddings_list)} embeddings locally.")
        except Exception as e:
            logger.error(f"Error generating embeddings with local model {model_name}: {e}", exc_info=True)
            return None
    elif invocation_method == 'remote_api':
        if not openai_client:
            logger.error("OpenAI client is not initialized for remote_api.")
            return None
        all_embeddings: List[Optional[List[float]]] = []
        batch_size = 50
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Remote Embeddings"):
            batch_texts = texts_to_embed[i:i + batch_size]
            try:
                response = openai_client.embeddings.create(model=model_name, input=batch_texts)
                if response.data:
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    if len(batch_embeddings) != len(batch_texts): logger.warning(f"API returned {len(batch_embeddings)} embeddings for {len(batch_texts)} texts in batch {i}.")
                else:
                     logger.warning(f"Remote API call for batch {i} returned no data.")
                     all_embeddings.extend([None] * len(batch_texts))
            except APIError as e:
                logger.error(f"API error during remote embedding for batch {i}: Status={e.status_code} Message={e.message}", exc_info=False)
                all_embeddings.extend([None] * len(batch_texts))
            except Exception as e:
                logger.error(f"Unexpected error during remote embedding for batch {i}: {e}", exc_info=True)
                all_embeddings.extend([None] * len(batch_texts))
        failed_count = sum(1 for emb in all_embeddings if emb is None)
        if failed_count > 0: logger.warning(f"{failed_count} remote embeddings failed.")
        else: logger.info(f"Successfully generated {len(all_embeddings)} embeddings via remote API.")
        embeddings_list = all_embeddings
    else:
        logger.error(f"Unsupported invocation method: {invocation_method}")
        return None
    return embeddings_list

def setup_chroma_vector_store(df: pd.DataFrame) -> Optional[chromadb.Collection]:
    # ... (代码不变, 增加距离度量配置) ...
    if not chromadb:
         logger.error("chromadb library is not available.")
         return None
    if 'embedding' not in df.columns:
        logger.error("DataFrame does not contain 'embedding' column.")
        return None
    logger.info("Setting up ChromaDB vector store...")
    try:
        chroma_client = chromadb.Client() # In-memory client
        # *** 修改：创建集合时指定距离度量为余弦相似度 ***
        collection = chroma_client.get_or_create_collection(
            CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # 指定使用 cosine 距离
        )
        logger.info(f"Using ChromaDB collection: '{CHROMA_COLLECTION_NAME}' with cosine distance.")

        ids_to_add, embeddings_to_add, documents_to_add, metadatas_to_add = [], [], [], []
        logger.info("Preparing data for ChromaDB...")
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Chroma Data"):
            if row['embedding'] is None or not isinstance(row['embedding'], list):
                 logger.warning(f"Skipping product ID {row['ID']} due to missing/invalid embedding.")
                 continue
            product_id_str = str(row['ID'])
            ids_to_add.append(product_id_str)
            embedding_floats = [float(x) for x in row['embedding']]
            embeddings_to_add.append(embedding_floats)
            name_str = str(row.get('Name', ''))
            desc_str = str(row.get('Description', ''))
            document_text = f"商品名称: {name_str}\n商品描述: {desc_str}"
            documents_to_add.append(document_text)
            metadata = { "original_id": int(row['ID']), "name": name_str, "price": float(row['Price']) if pd.notna(row['Price']) else 0.0, "stock": int(row['Stock']) if pd.notna(row['Stock']) else 0 }
            metadatas_to_add.append(metadata)
        if not ids_to_add:
             logger.error("No valid data to add to ChromaDB.")
             return None
        logger.info(f"Adding {len(ids_to_add)} items to ChromaDB collection...")
        collection.add(ids=ids_to_add, embeddings=embeddings_to_add, documents=documents_to_add, metadatas=metadatas_to_add)
        item_count = collection.count()
        logger.info(f"Successfully added/updated data. Collection '{CHROMA_COLLECTION_NAME}' now contains {item_count} items.")
        return collection
    except Exception as e:
        logger.error(f"Error setting up or adding data to ChromaDB: {e}", exc_info=True)
        return None

# --- generate_single_embedding 函数 (保持不变) ---
def generate_single_embedding(text: str, model_name: str, invocation_method: str) -> Optional[List[float]]:
    # ... (代码不变) ...
    if not text: return None
    if invocation_method == 'local':
        model = _load_local_embedding_model(model_name)
        if not model: return None
        try:
            embedding = model.encode(text)
            return embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        except Exception as e:
            logger.error(f"Error generating single local embedding: {e}", exc_info=True)
            return None
    elif invocation_method == 'remote_api':
        if not openai_client:
            logger.error("OpenAI client not initialized for single remote embedding.")
            return None
        try:
            response = openai_client.embeddings.create(model=model_name, input=[text])
            if response.data and response.data[0].embedding: return response.data[0].embedding
            else: logger.error(f"Remote API call for single text failed: {response}"); return None
        except APIError as e:
            logger.error(f"API error during single remote embedding: Status={e.status_code} Message={e.message}", exc_info=False); return None
        except Exception as e:
            logger.error(f"Unexpected error during single remote embedding: {e}", exc_info=True); return None
    else:
        logger.error(f"Unsupported invocation method: {invocation_method}"); return None

# --- retrieve_relevant_docs 函数 (保持不变, 但现在 distance 是 cosine distance) ---
def retrieve_relevant_docs(
    question_text: str, collection: chromadb.Collection, embedding_model_name: str,
    invocation_method: str, top_n: int = 3
) -> Optional[List[Dict[str, Any]]]:
    # ... (代码不变, 但返回的 distance 现在是 cosine distance) ...
    if not collection: logger.error("ChromaDB collection unavailable."); return None
    logger.info(f"\nRetrieving documents for question: '{question_text}'")
    question_embedding = generate_single_embedding(question_text, embedding_model_name, invocation_method)
    if question_embedding is None: logger.error("Failed to generate question embedding."); return None
    try:
        results: QueryResult = collection.query(
            query_embeddings=[question_embedding], n_results=top_n, include=['documents', 'metadatas', 'distances']
        )
        logger.debug(f"ChromaDB query results: {results}")
        retrieved_docs = []
        if results and results.get('ids') and len(results['ids']) > 0:
            ids, documents, metadatas, distances = results['ids'][0], results.get('documents', [[]])[0], results.get('metadatas', [[]])[0], results.get('distances', [[]])[0]
            for i in range(len(ids)):
                # 对于 cosine 距离, similarity = 1 - distance
                distance = distances[i] if distances else -1.0
                similarity = 1.0 - distance if distance >= 0 else 0.0
                doc_info = {
                    "id": ids[i],
                    "document": documents[i] if documents else "N/A",
                    "metadata": metadatas[i] if metadatas else {},
                    "distance": distance,
                    "similarity": similarity
                }
                doc_info.update(doc_info.get("metadata", {}))
                retrieved_docs.append(doc_info)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        else:
            logger.info("No relevant documents found."); return []
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True); return None

# --- 新增函数：构建 RAG Prompt ---
def build_rag_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """构建包含检索到的上下文的 Prompt"""
    context = "--- 相关商品信息 ---\n"
    if not retrieved_docs:
        context += "(未找到相关商品信息)\n"
    else:
        for i, doc in enumerate(retrieved_docs):
            context += f"商品 {i+1}:\n"
            context += f"  名称: {doc.get('name', 'N/A')}\n"
            context += f"  描述: {doc.get('document', 'N/A').split('商品描述: ')[-1][:200]}...\n" # 提取描述部分并截断
            context += f"  价格: {doc.get('price', 'N/A'):.2f}\n"
            context += f"  库存: {doc.get('stock', 'N/A')}\n"
            context += f"  (相似度: {doc.get('similarity', 0.0):.4f})\n\n"
    context += "--- 相关商品信息结束 ---\n\n"

    instruction = (
        "你是一个电商问答助手。\n"
        "请根据下面提供的“相关商品信息”来回答用户的问题。\n"
        "请主要依据这些信息进行回答，保持回答简洁、相关。\n"
        "如果提供的信息不足以回答问题，请直接说明“根据我找到的相关信息，无法直接回答您的问题。”\n"
        "不要编造信息中不存在的内容。"
    )

    final_prompt = f"{instruction}\n\n{context}用户的问题是：\n{question}\n\n请回答："
    return final_prompt

# --- 新增函数：调用 LLM 获取 RAG 回答 ---
def get_llm_rag_response(prompt: str, chat_model_name: str) -> str:
    """使用指定的对话模型获取 RAG 回答"""
    if not openai_client:
        return "错误：OpenAI 客户端未初始化，无法获取回答。"
    if not chat_model_name:
        return "错误：未指定有效的对话模型名称。"

    logger.info(f"Sending prompt to LLM ({chat_model_name})... Prompt length: {len(prompt)}")
    logger.debug(f"LLM Prompt (first 300 chars): {prompt[:300]}...")

    try:
        # 使用 Chat Completion API
        response = openai_client.chat.completions.create(
            model=chat_model_name,
            messages=[
                # 可以将整个 prompt 作为 system message 或 user message
                # 这里作为 system message 包含指令和上下文，user message 是原始问题
                # 或者简单地将整个 final_prompt 作为 user message
                # {"role": "system", "content": prompt}, # <--- 方式一
                 {"role": "user", "content": prompt}      # <--- 方式二 (更简单)
            ],
            max_tokens=500, # 限制回答长度
            temperature=0.3 # 降低随机性，使其更关注上下文
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM ({chat_model_name}) generated answer (first 100 chars): {answer[:100]}...")
            return answer
        else:
            logger.error(f"LLM ({chat_model_name}) returned an unexpected response structure: {response}")
            return "抱歉，AI 模型返回了意外的响应格式。"

    except RateLimitError as e:
        logger.error(f"LLM ({chat_model_name}) Rate Limit Error: {e}. Waiting and retrying once...")
        try:
             time.sleep(5) # 等待 5 秒
             response = openai_client.chat.completions.create(model=chat_model_name, messages=[{"role": "user", "content": prompt}], max_tokens=500, temperature=0.3)
             if response.choices and response.choices[0].message and response.choices[0].message.content: return response.choices[0].message.content.strip()
             else: return "抱歉，重试后 AI 模型仍返回意外格式。"
        except Exception as retry_e:
             logger.error(f"LLM ({chat_model_name}) Retry Error: {retry_e}")
             return f"抱歉，尝试从速率限制中恢复失败：{retry_e}"
    except APIConnectionError as e:
         logger.error(f"LLM ({chat_model_name}) Connection Error: {e}")
         return f"抱歉，无法连接到 AI 模型服务：{e}"
    except AuthenticationError as e:
        logger.error(f"LLM ({chat_model_name}) Authentication Error: {e}")
        return f"抱歉，AI 模型服务认证失败，请检查 API Key。 ({e})"
    except Exception as e:
        logger.error(f"LLM ({chat_model_name}) Unexpected Error: {e}", exc_info=True)
        return f"抱歉，与 AI 模型交互时发生未知错误：{e}"

# --- 主执行流程 ---
if __name__ == "__main__":
    logger.info("--- Starting RAG Simulation Script ---")

    # 1. 加载环境变量并初始化 Client
    if not load_environment_variables():
        logger.critical("Failed to load environment or initialize OpenAI client. Exiting.")
        exit() # 如果 Client 初始化失败，则退出

    # 2. 加载商品数据
    product_df = load_product_data(CSV_FILE_PATH)

    # --- 初始化变量 ---
    product_collection = None
    generated_embeddings = None
    rag_results = {} # 存储最终结果

    if product_df is not None:
        logger.info("Product data loaded successfully.")
        # ... (打印 head 的代码) ...
        try: print(product_df.head().to_markdown(index=False))
        except ImportError: print(product_df.head())

        # --- 步骤 3 (Skipped) & 4 ---
        logger.info("\n--- Step 3: Text Chunking (Skipped) ---")
        logger.info(f"\n--- Step 4: Generating Embeddings...")
        generated_embeddings = generate_embeddings_for_products(
            product_df, ACTIVE_EMBEDDING_MODEL_NAME, ACTIVE_EMBEDDING_INVOCATION_METHOD
        )

        if generated_embeddings and len(generated_embeddings) == len(product_df):
            # ... (处理 Embedding 结果) ...
            successful_embeddings = [emb for emb in generated_embeddings if emb is not None]
            failed_count = len(generated_embeddings) - len(successful_embeddings)
            logger.info(f"Embedding generation finished. Success: {len(successful_embeddings)}, Failed: {failed_count}")
            product_df['embedding'] = generated_embeddings

            if successful_embeddings:
                first_valid_embedding = np.array(successful_embeddings[0])
                logger.info(f"Shape/Dimension of the first generated embedding: {first_valid_embedding.shape}")

                # --- 步骤 5: 设置向量存储 ---
                logger.info("\n--- Step 5: Setting up Vector Store ---")
                # *** 添加了修改，指定 cosine 距离 ***
                product_collection = setup_chroma_vector_store(product_df)

                if product_collection:
                    logger.info("Vector store setup completed successfully.")
                else:
                    logger.error("Failed to setup vector store.")
            else:
                logger.warning("No valid embeddings generated, skipping vector store setup.")
        # ... (处理 Embedding 失败或数量不匹配) ...
        elif generated_embeddings is None: logger.error("Embedding generation function returned None.")
        else: logger.error(f"Embedding generation returned unexpected results.")
    else:
        logger.error("Failed to load product data.")

    # --- 执行 RAG 流程 (仅当向量库成功设置) ---
    if product_collection:
        # --- 步骤 6: 定义测试问题 ---
        logger.info("\n--- Step 6: Defining Test Questions ---")
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

        # --- 步骤 7-10: 检索、构建 Prompt、调用 LLM 并记录结果 ---
        logger.info("\n--- Steps 7-10: Running RAG Pipeline ---")

        for question in test_questions:
            print(f"\n{'='*10} Processing Question: {question} {'='*10}")

            # Step 7: Retrieve
            retrieved_docs = retrieve_relevant_docs(
                question, product_collection, ACTIVE_EMBEDDING_MODEL_NAME,
                ACTIVE_EMBEDDING_INVOCATION_METHOD, top_n=RETRIEVAL_TOP_N
            )

            if retrieved_docs is None: # 检索过程中出错
                print("  Retrieval failed.")
                rag_results[question] = {"retrieved": "ERROR", "prompt": "N/A", "answer": "ERROR"}
                continue
            elif not retrieved_docs: # 没找到相关文档
                print("  No relevant documents found.")
                # 仍然可以尝试让 LLM 回答，但不提供上下文
                # context_for_prompt = []
            # else: # 找到了文档
            #     context_for_prompt = retrieved_docs

            # Step 8: Build Prompt
            final_prompt = build_rag_prompt(question, retrieved_docs if retrieved_docs else [])
            # 打印检索到的上下文（用于对比）
            print("\n  --- Retrieved Context (Formatted for Prompt) ---")
            context_part = final_prompt.split("--- 相关商品信息 ---\n")[1].split("--- 相关商品信息结束 ---\n")[0]
            print(context_part.strip())
            print("  --- End Retrieved Context ---")

            # Step 9: Get LLM Response
            llm_answer = get_llm_rag_response(final_prompt, ACTIVE_CHAT_MODEL_NAME)

            # Step 10: Record Result
            rag_results[question] = {
                "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                "top_retrieved_names": [doc.get('name') for doc in retrieved_docs[:RETRIEVAL_TOP_N]] if retrieved_docs else [],
                # "prompt": final_prompt, # Prompt 可能很长，选择性记录
                "answer": llm_answer
            }
            print(f"\n  --- LLM Answer ---")
            print(f"  {llm_answer}")
            print(f"  --- End LLM Answer ---")
            print(f"{'='*40}")
            time.sleep(1) # 防止过于频繁地调用 LLM API (如果免费额度有限)

        # --- (可选) 打印最终 RAG 结果概览 ---
        # logger.info("\n--- Final RAG Results Summary ---")
        # for q, res in rag_results.items():
        #     logger.info(f"Q: {q}")
        #     logger.info(f"  Retrieved: {res['retrieved_count']} items - Top: {res['top_retrieved_names']}")
        #     logger.info(f"  Answer: {res['answer']}")
        # logger.info("-" * 30)

    else: # product_collection 未成功创建
         logger.warning("Skipping RAG pipeline due to issues in previous stages.")

    logger.info("--- RAG Simulation Script Finished ---")