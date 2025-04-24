# app/routes/qa.py (动态模型 + 多轮对话 + 上下文增强 v5)

from flask import Blueprint, request, jsonify
# 导入需要用到的服务函数和模型
from ..services.openai_service import get_openai_response
from ..services.data_service import get_all_rules, get_rules_by_category
# --- 修改：导入 AiModel ---
from ..models import Product, AiModel # <--- 添加 AiModel
import logging

qa = Blueprint('qa', __name__)
logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10
# --- 新增：定义一个后备/默认的模型名称 ---
DEFAULT_FALLBACK_MODEL = "gpt-4.1" # 或者 'gpt-3.5-turbo'

@qa.route('/ask', methods=['POST'])
def ask_question():
    """
    (最终版) 接收问题和历史记录，根据上下文构建消息列表，
    查询当前激活的 AI 模型，调用 OpenAI 服务并返回答案和更新后的历史记录。
    """
    logger.info("--- 收到 /ask 请求 (动态模型版 v5) ---")

    if not request.is_json:
        logger.error("!!! 请求不是 JSON 格式 !!!")
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    user_question = data.get('question')
    product_id = data.get('product_id')
    context_type = data.get('context_type')
    conversation_history = data.get('history')
    if conversation_history is None or not isinstance(conversation_history, list):
         conversation_history = []

    if not user_question:
        logger.error("!!! 请求中缺少 'question' 字段 !!!")
        return jsonify({"error": "请求中缺少 'question' 字段"}), 400

    # --- 1. 构建上下文信息 (逻辑不变) ---
    context_info = ""
    base_system_instruction = "你是一个友好的电商问答助手。"
    specific_instruction = ""
    is_product_context = False
    # ...(场景判断和 context_info 构建逻辑保持不变)...
    # 场景 0: 商家帮助
    if context_type == 'seller_help':
        logger.info("--- 处理商家帮助场景 ---")
        specific_instruction = (f"你是一位经验丰富的电商平台商家后台操作助手。\n" f"请根据以下提供的 **商家操作指南** 回答用户（商家）在后台操作时遇到的问题。\n" f"回答应尽可能清晰、步骤化。如果信息不足，请说明根据现有指南无法回答该操作问题。")
        seller_rules = get_rules_by_category('商家操作')
        context_info += "--- 商家操作指南 ---\n"
        if seller_rules:
            for rule in seller_rules: context_info += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
        else: context_info += "（抱歉，暂无相关的商家操作指南信息）\n"
        context_info += "--- 指南结束 ---\n"
    # 场景 1: 商品详情
    elif product_id:
        logger.info(f"--- 处理商品详情场景 (ID: {product_id}) ---")
        try:
            product = Product.query.get(int(product_id))
            if product:
                is_product_context = True; logger.info(f"--- 成功找到商品: {product.name} ---")
                product_context_str = f"--- 关于商品 '{product.name}' (ID: {product.id}) 的信息 ---\n"; product_context_str += f"描述: {product.description or '暂无描述'}\n"; product_context_str += f"价格: {product.price:.2f} 元\n"; product_context_str += f"当前库存: {product.stock} 件\n"; product_context_str += f"由卖家 '{product.seller.username}' 提供\n"; product_context_str += "--- 商品信息结束 ---\n\n"
                context_info += product_context_str
                specific_instruction = (f"用户正在查看商品 '{product.name}'。请主要根据以上提供的 **特定商品信息** 来回答用户的问题。\n" f"如果商品信息不足以回答，可以参考下面补充的通用平台规则。\n" f"如果根据所有信息都无法回答，请说明“根据现有信息无法回答关于此商品的问题”。")
                all_rules = get_all_rules(); general_context = "\n--- (补充) 平台规则与常见问题解答 ---\n"
                if all_rules:
                    for rule in all_rules: general_context += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
                else: general_context += "（抱歉，目前没有可参考的通用规则信息）\n"
                general_context += "--- 平台信息结束 ---\n"; context_info += general_context
            else: logger.warning(f"!!! 数据库中未找到 ID 为 {product_id} 的商品，按通用场景处理 !!!")
        except Exception as e: logger.error(f"!!! 查询商品 (ID: {product_id}) 时出错: {e} !!!", exc_info=True)
    # 场景 2: 通用提问
    if not context_type and not is_product_context:
        logger.info("--- 处理通用规则场景 ---")
        specific_instruction = (f"请仔细阅读以下关于我们平台的信息，并仅根据这些信息回答用户的问题。\n" f"如果信息不足以回答，请直接说明“根据我所掌握的信息，无法回答这个问题”。")
        all_rules = get_all_rules(); general_context = "--- 平台规则与常见问题解答 ---\n"
        if all_rules:
            for rule in all_rules: general_context += f"Q: {rule.get('question', 'N/A')}\nA: {rule.get('answer', 'N/A')}\n\n"
        else: general_context += "（抱歉，目前没有可参考的规则信息）\n"
        general_context += "--- 平台信息结束 ---\n"; context_info += general_context

    # --- 2. 构建 System Prompt (逻辑不变) ---
    system_prompt = f"{base_system_instruction}\n{specific_instruction}\n\n"
    if context_info: system_prompt += f"请参考以下信息来回答：\n{context_info}"
    else: system_prompt += "（当前无特定上下文信息）"

    # --- 3. 构建消息列表 (逻辑不变) ---
    messages_to_send = []; messages_to_send.append({"role": "system", "content": system_prompt})
    start_index = max(0, len(conversation_history) - MAX_HISTORY_MESSAGES)
    messages_to_send.extend(conversation_history[start_index:])
    if len(conversation_history) > MAX_HISTORY_MESSAGES: logger.warning(f"--- 对话历史过长 ({len(conversation_history)} 条)，已截断为最后 {MAX_HISTORY_MESSAGES} 条 ---")
    messages_to_send.append({"role": "user", "content": user_question})
    # logger.debug(f"--- [qa.py] 构建的最终 messages_to_send (共 {len(messages_to_send)} 条): {messages_to_send} ---")

    # --- 4. 查询当前激活的 AI 模型 ---
    active_model_name = None
    try:
        active_model = AiModel.query.filter_by(is_active=True).first()
        if active_model:
            active_model_name = active_model.model_name
            logger.info(f"--- 使用当前激活的 AI 模型: {active_model_name} ---")
        else:
            logger.warning(f"--- 未找到激活的 AI 模型，将使用默认后备模型: {DEFAULT_FALLBACK_MODEL} ---")
            active_model_name = DEFAULT_FALLBACK_MODEL
    except Exception as e:
        logger.error(f"!!! 查询激活 AI 模型时出错: {e}，将使用默认后备模型: {DEFAULT_FALLBACK_MODEL} !!!", exc_info=True)
        active_model_name = DEFAULT_FALLBACK_MODEL

    # --- 5. 调用 AI 服务 ---
    # model_to_use = "gpt-4.1" # <--- 移除硬编码
    answer = get_openai_response(messages=messages_to_send, model=active_model_name) # <--- 传递查询到的模型名称

    # --- 6. 更新历史记录 (逻辑不变) ---
    updated_history = conversation_history
    updated_history.append({"role": "user", "content": user_question})
    if not answer.startswith("抱歉"):
        updated_history.append({"role": "assistant", "content": answer})
    updated_history = updated_history[max(0, len(updated_history) - MAX_HISTORY_MESSAGES):]
    logger.info(f"--- 返回的答案 (前 100 字符): {answer[:100]}... ---")
    logger.info(f"--- 更新后的历史记录包含 {len(updated_history)} 条消息 ---")

    # --- 7. 返回 JSON 响应 (逻辑不变) ---
    return jsonify({
        "answer": answer,
        "history": updated_history
    })