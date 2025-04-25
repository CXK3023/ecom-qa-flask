# app/routes/qa.py (Add Price/Stock to Vector Search Context v8.2)

from flask import Blueprint, request, jsonify
# 导入服务
from ..services.openai_service import get_openai_response
from ..services.data_service import get_all_rules, get_rules_by_category
from ..services.embedding_service import (
    generate_embedding,
    deserialize_embedding,
    calculate_cosine_distance
)
# --- 导入模型 ---
from ..models import Product, AiModel, EmbeddingModel, SystemSetting
from .. import db
import logging
import numpy as np
from scipy.spatial.distance import cdist # 确保导入

qa = Blueprint('qa', __name__)
logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10
DEFAULT_CHAT_FALLBACK_MODEL = "gpt-4.1"
VECTOR_SEARCH_TOP_N = 3
VECTOR_SEARCH_DISTANCE_THRESHOLD = 0.8
DEFAULT_ENABLE_VECTOR_SEARCH = False

@qa.route('/ask', methods=['POST'])
def ask_question():
    """
    (最终版 v8.2 - 向量搜索上下文添加价格库存)
    """
    logger.info("--- 收到 /ask 请求 (向量上下文增强 v8.2) ---")

    if not request.is_json: # ... (请求校验不变) ...
        return jsonify({"error": "请求必须是 JSON 格式"}), 400
    data = request.get_json()
    user_question = data.get('question')
    # ... (获取其他参数不变) ...
    product_id = data.get('product_id')
    context_type = data.get('context_type')
    conversation_history = data.get('history', [])
    if not user_question: return jsonify({"error": "请求中缺少 'question' 字段"}), 400

    # --- 0. 查询活动模型 和 全局设置 (逻辑不变) ---
    active_chat_model_name = DEFAULT_CHAT_FALLBACK_MODEL
    active_embedding_model = None
    vector_search_globally_enabled = DEFAULT_ENABLE_VECTOR_SEARCH
    try:
        active_chat_model = AiModel.query.filter_by(is_active=True).first()
        if active_chat_model: active_chat_model_name = active_chat_model.model_name
        # ... (查询逻辑不变) ...
        else: logger.warning(f"未找到活动对话模型，使用后备: {DEFAULT_CHAT_FALLBACK_MODEL}")
        logger.info(f"活动对话模型: {active_chat_model_name}")

        active_embedding_model = EmbeddingModel.query.filter_by(is_active=True).first()
        if active_embedding_model: logger.info(f"活动向量模型: {active_embedding_model.model_name} (Method: {active_embedding_model.invocation_method})")
        else: logger.info("未找到活动向量模型。")

        setting = SystemSetting.query.filter_by(key='enable_vector_search').first()
        if setting and setting.value == 'true': vector_search_globally_enabled = True
        logger.info(f"全局向量搜索设置: {'启用' if vector_search_globally_enabled else '禁用'}")
    except Exception as e:
        logger.error(f"!!! 查询活动模型或设置时出错: {e}. 使用后备设置 !!!", exc_info=True)


    # --- 1. 构建上下文信息 (逻辑不变) ---
    context_info = ""
    base_system_instruction = "你是一个友好的电商问答助手。"
    specific_instruction = ""
    relevant_products_context = ""
    is_product_context = False
    is_seller_context = False

    # 场景 0: 商家帮助 (逻辑不变)
    if context_type == 'seller_help':
        is_seller_context = True
        # ... (商家场景代码不变) ...
        logger.info("--- 处理商家帮助场景 ---")
        specific_instruction = (f"你是一位经验丰富的电商平台商家后台操作助手。\n...")
        seller_rules = get_rules_by_category('商家操作'); context_info += "--- 商家操作指南 ---\n"
        if seller_rules: # ... (规则构建不变) ...
             for rule in seller_rules: context_info += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
        else: context_info += "（抱歉，暂无相关的商家操作指南信息）\n"; context_info += "--- 指南结束 ---\n"

    # 场景 1: 商品详情页 (逻辑不变)
    elif product_id:
        # ... (商品详情场景代码不变) ...
        logger.info(f"--- 处理商品详情场景 (ID: {product_id}) ---")
        try:
            product = Product.query.get(int(product_id))
            if product:
                is_product_context = True; logger.info(f"--- 成功找到商品: {product.name} ---")
                product_context_str = f"--- 关于商品 '{product.name}' (ID: {product.id}) 的信息 ---\n"
                product_context_str += f"描述: {product.description or '暂无描述'}\n"
                product_context_str += f"价格: {product.price:.2f} 元\n" # <-- 已包含价格库存
                product_context_str += f"当前库存: {product.stock} 件\n" # <-- 已包含价格库存
                if product.seller: product_context_str += f"由卖家 '{product.seller.username}' 提供\n"
                product_context_str += "--- 商品信息结束 ---\n\n"
                context_info += product_context_str
                specific_instruction = (f"用户正在查看商品 '{product.name}'。请主要根据以上提供的 **特定商品信息** 来回答用户的问题。\n...")
                all_rules = get_all_rules(); general_context = "\n--- (补充) 平台规则与常见问题解答 ---\n"
                if all_rules: # ... (规则构建不变) ...
                     for rule in all_rules: general_context += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
                else: general_context += "（抱歉，目前没有可参考的通用规则信息）\n"; general_context += "--- 平台信息结束 ---\n"
                context_info += general_context
            else: logger.warning(f"!!! 未找到商品 {product_id} !!!")
        except Exception as e: logger.error(f"!!! 查询商品 {product_id} 出错: {e} !!!", exc_info=True)


    # 场景 2: 通用提问
    if not is_seller_context and not is_product_context:
        logger.info("--- 处理通用问答场景 ---")
        if vector_search_globally_enabled and active_embedding_model:
            logger.info(f"--- 开始向量搜索 (全局开关开启, 活动模型: {active_embedding_model.model_name}) ---")
            question_embedding = None
            try:
                # 1. 生成问题向量 (逻辑不变)
                serialized_question_embedding = generate_embedding(user_question)
                if serialized_question_embedding: question_embedding = deserialize_embedding(serialized_question_embedding)

                if question_embedding is None: logger.error("!!! 问题向量生成失败，无法执行向量搜索 !!!")
                else:
                    # 2. 查询带向量的商品 (逻辑不变)
                    products_with_embeddings = Product.query.filter(Product.embedding.isnot(None)).all()
                    if products_with_embeddings:
                        product_embeddings = []; valid_products = []
                        for p in products_with_embeddings: # 反序列化 (逻辑不变)
                            vec = deserialize_embedding(p.embedding)
                            if vec is not None: product_embeddings.append(vec); valid_products.append(p)
                            else: logger.warning(f"反序列化商品 {p.id} 向量失败。")

                        if product_embeddings:
                            # 3. 计算距离 & 找 Top N (逻辑不变)
                            all_product_embeddings_np = np.array(product_embeddings)
                            if isinstance(question_embedding, np.ndarray):
                                distances = cdist(question_embedding.reshape(1, -1), all_product_embeddings_np, 'cosine')[0]
                                top_n_indices = np.argsort(distances)[:VECTOR_SEARCH_TOP_N]

                                # === vvv 修改：构建上下文时加入价格和库存 vvv ===
                                relevant_products_context += f"\n--- 可能相关的商品 (Top {VECTOR_SEARCH_TOP_N}) ---\n"
                                found_relevant = False
                                for index in top_n_indices:
                                    if index < len(valid_products) and distances[index] < VECTOR_SEARCH_DISTANCE_THRESHOLD:
                                        relevant_product = valid_products[index]
                                        dist = distances[index]
                                        relevant_products_context += (
                                            f"商品: {relevant_product.name} (ID: {relevant_product.id}, 相关度评分: {(1-dist):.2f})\n"
                                            # 添加价格和库存信息
                                            f"价格: {relevant_product.price:.2f} 元, 库存: {relevant_product.stock} 件\n"
                                            f"描述: {relevant_product.description or '无'[:50]}...\n\n"
                                        )
                                        found_relevant = True
                                if not found_relevant: relevant_products_context += "(未找到高度相关的商品)\n"
                                relevant_products_context += "--- 相关商品信息结束 ---\n"
                                # === ^^^ 修改结束 ^^^ ===
                                logger.info("--- 向量搜索完成，已构建相关商品上下文 ---")
                            else: logger.error("!!! 问题向量不是有效的 NumPy 数组，无法计算距离 !!!")
                        else: logger.warning("--- 没有有效的商品向量可供比较 ---")
                    else: logger.info("--- 数据库中没有带向量的商品 ---")
            except Exception as e:
                logger.error(f"!!! 向量搜索过程中发生错误: {e} !!!", exc_info=True)
                relevant_products_context = "\n(向量搜索时遇到问题)\n"
        else: # ... (向量搜索禁用日志不变) ...
             if not vector_search_globally_enabled: logger.info("--- 向量搜索已在系统设置中禁用 ---")
             elif not active_embedding_model: logger.info("--- 向量搜索已禁用 (无活动向量模型) ---")

        # --- 构建通用场景指令和基础规则上下文 (逻辑不变) ---
        specific_instruction = (f"请仔细阅读以下平台规则{(relevant_products_context and '和可能相关的商品信息') or ''}，并据此回答用户的问题。\n...")
        all_rules = get_all_rules(); general_context = "\n--- 平台规则与常见问题解答 ---\n"
        if all_rules: # ... (规则构建不变) ...
             for rule in all_rules: general_context += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
        else: general_context += "（暂无规则信息）\n"; general_context += "--- 平台信息结束 ---\n"
        context_info = relevant_products_context + general_context


    # --- 2. 构建最终 System Prompt (逻辑不变) ---
    system_prompt = f"{base_system_instruction}\n{specific_instruction}\n\n"
    if context_info: system_prompt += f"请参考以下信息来回答：\n{context_info}"
    else: system_prompt += "（当前无特定上下文信息）"

    # --- 3. 构建消息列表 (逻辑不变) ---
    messages_to_send = [{"role": "system", "content": system_prompt}]
    # ... (历史记录处理不变) ...
    start_index = max(0, len(conversation_history) - MAX_HISTORY_MESSAGES)
    messages_to_send.extend(conversation_history[start_index:])
    if len(conversation_history) > MAX_HISTORY_MESSAGES: logger.warning(f"对话历史过长，已截断")
    messages_to_send.append({"role": "user", "content": user_question})

    # --- 4. 调用 AI 服务 (逻辑不变) ---
    answer = get_openai_response(messages=messages_to_send, model=active_chat_model_name)

    # --- 5. 更新历史记录 (逻辑不变) ---
    updated_history = conversation_history
    # ... (历史记录更新不变) ...
    updated_history.append({"role": "user", "content": user_question})
    if answer and not answer.startswith("抱歉") and not answer.startswith("错误"):
        updated_history.append({"role": "assistant", "content": answer})
    updated_history = updated_history[max(0, len(updated_history) - MAX_HISTORY_MESSAGES):]
    logger.info(f"--- 返回的答案 (前 100 字符): {answer[:100]}... ---")

    # --- 6. 返回 JSON 响应 (逻辑不变) ---
    return jsonify({ "answer": answer, "history": updated_history })