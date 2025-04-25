# app/services/embedding_service.py (Refactored for EmbeddingModel)
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist # 用于计算余弦距离
import logging
from typing import Optional, Tuple # 导入 Tuple 用于类型提示
# --- 修改：导入 EmbeddingModel ---
from ..models import EmbeddingModel # <--- 导入新的 EmbeddingModel
from .. import db # 导入数据库实例
# 导入 OpenAI 客户端实例和可能的错误类型
from .openai_service import client as openai_client, AuthenticationError, APIConnectionError, RateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 本地 SentenceTransformer 模型缓存
local_model_cache = {}
DEFAULT_LOCAL_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # 默认本地 Embedding 模型

# --- 修改：查询 EmbeddingModel ---
def get_active_embedding_model_info() -> Tuple[Optional[str], Optional[str]]:
    """
    从数据库获取当前激活的 Embedding 模型信息。

    Returns:
        Tuple[Optional[str], Optional[str]]: 返回 (model_name, invocation_method) 元组。
                                             如果找不到活动模型或出错，则返回 (None, None) 或默认值。
                                             当前实现：找不到则回退到默认本地模型。
    """
    try:
        # 查询活动的 EmbeddingModel
        active_model = EmbeddingModel.query.filter_by(is_active=True).first()
        if active_model:
            logger.info(f"Found active EmbeddingModel: {active_model.model_name} (Method: {active_model.invocation_method})")
            return active_model.model_name, active_model.invocation_method
        else:
            logger.warning(f"No active EmbeddingModel found in DB. Falling back to default local model: {DEFAULT_LOCAL_EMBEDDING_MODEL_NAME}")
            # 回退到默认本地模型配置
            return DEFAULT_LOCAL_EMBEDDING_MODEL_NAME, 'local'
    except Exception as e:
        logger.error(f"Error fetching active EmbeddingModel from DB: {e}. Falling back to default local model.", exc_info=True)
        # 出错也回退到默认本地模型配置
        return DEFAULT_LOCAL_EMBEDDING_MODEL_NAME, 'local'

def _load_local_model(model_name: str) -> Optional[SentenceTransformer]:
    """加载本地 SentenceTransformer 模型，使用缓存 (逻辑不变)"""
    global local_model_cache
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

def generate_embedding(text: str) -> Optional[bytes]:
    """
    根据当前激活的 Embedding 模型配置生成文本的向量嵌入。
    (逻辑不变，但依赖 get_active_embedding_model_info 的新实现)
    """
    if not text or not isinstance(text, str):
        logger.error("Invalid input text provided for embedding generation.")
        return None

    model_name, invocation_method = get_active_embedding_model_info()

    # --- 添加检查：如果连默认模型信息都无法获取 ---
    if not model_name or not invocation_method:
        logger.error("Could not determine active embedding model configuration. Cannot generate embedding.")
        return None
    # --- 检查结束 ---

    logger.info(f"Generating embedding using model '{model_name}' via '{invocation_method}' method.")
    vector = None

    if invocation_method == 'local':
        model = _load_local_model(model_name)
        if model:
            try:
                vector = model.encode(text)
                logger.info(f"Successfully generated embedding locally using {model_name}.")
            except Exception as e:
                logger.error(f"Error generating embedding with local model {model_name}: {e}", exc_info=True)
        else:
            logger.error(f"Local model {model_name} could not be loaded. Cannot generate embedding.")

    elif invocation_method == 'remote_api':
        if not openai_client:
            logger.error("OpenAI client (for remote API) is not initialized. Cannot generate embedding.")
            return None
        try:
            response = openai_client.embeddings.create(
                model=model_name, # 使用从 EmbeddingModel 获取的模型名
                input=[text]
            )
            if response.data and response.data[0].embedding:
                vector = np.array(response.data[0].embedding)
                logger.info(f"Successfully generated embedding via remote API using {model_name}.")
            else:
                logger.error(f"Remote API call for embedding using {model_name} returned unexpected data: {response}")
        except (AuthenticationError, APIConnectionError, RateLimitError) as e:
             logger.error(f"API error during remote embedding generation ({type(e).__name__}) using {model_name}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during remote embedding generation using {model_name}: {e}", exc_info=True)

    else:
        logger.error(f"Unsupported invocation method: {invocation_method}")

    # 序列化向量 (逻辑不变)
    if vector is not None:
        try:
            serialized_vector = pickle.dumps(vector)
            return serialized_vector
        except (pickle.PicklingError, TypeError) as e:
            logger.error(f"Error serializing the generated vector: {e}", exc_info=True)
            return None
    else:
        logger.error("Embedding generation failed.")
        return None

def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
    """将存储的二进制数据反序列化为 NumPy 向量 (逻辑不变)"""
    if not serialized_embedding: return None
    try:
        vector = pickle.loads(serialized_embedding)
        return vector if isinstance(vector, np.ndarray) else None
    except Exception as e:
        logger.error(f"Error deserializing embedding: {e}", exc_info=True)
        return None

def calculate_cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个 NumPy 向量之间的余弦距离 (逻辑不变)"""
    if vec1.ndim == 1: vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1: vec2 = vec2.reshape(1, -1)
    try:
        distance = cdist(vec1, vec2, 'cosine')[0, 0]
        return float(distance) if not np.isnan(distance) else float('inf')
    except Exception as e:
        logger.error(f"Error calculating cosine distance: {e}", exc_info=True)
        return float('inf')

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个 NumPy 向量之间的余弦相似度 (逻辑不变)"""
    distance = calculate_cosine_distance(vec1, vec2)
    if distance == float('inf'): return -1.0
    similarity = 1.0 - distance
    return max(-1.0, min(1.0, similarity))