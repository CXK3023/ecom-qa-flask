# app/routes/qa.py (用下面的代码替换整个 ask_question 函数 - Day 7 最终版)

from flask import Blueprint, request, jsonify
# 导入需要用到的服务函数和模型
from ..services.openai_service import get_openai_response
from ..services.data_service import get_all_rules, get_rules_by_category # 导入两个函数
from ..models import Product # 导入 Product

qa = Blueprint('qa', __name__)

@qa.route('/ask', methods=['POST'])
def ask_question():
    """
    接收问题，根据上下文(商家帮助/商品ID/通用)构建 Prompt，
    调用 OpenAI 服务并返回答案。
    """
    print("--- 收到 /ask 请求 (上下文增强版 v3 - 场景感知) ---")

    if not request.is_json:
        print("!!! [qa.py] 请求不是 JSON 格式 !!!")
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    user_question = data.get('question')
    product_id = data.get('product_id')
    # === 新增：获取上下文类型 ===
    context_type = data.get('context_type') # 例如 'seller_help'

    print(f"--- [qa.py] 用户问题: {user_question} ---")
    print(f"--- [qa.py] 商品ID: {product_id} ---")
    print(f"--- [qa.py] 上下文类型: {context_type} ---") # 打印收到的类型

    if not user_question:
        print("!!! [qa.py] 请求中缺少 'question' 字段 !!!")
        return jsonify({"error": "请求中缺少 'question' 字段"}), 400

    # --- 根据场景优先级确定上下文和指令 ---
    context = ""
    prompt_instruction = ""
    is_product_context = False # 重置标记

    # === 逻辑优先级：商家帮助 -> 商品详情 -> 通用 ===

    # === 场景 0: 商家帮助 ===
    if context_type == 'seller_help':
        print("--- [qa.py] 处理商家帮助场景 ---")
        prompt_instruction = (
            f"你是一位经验丰富的电商平台商家后台操作助手。\n"
            f"请根据以下提供的 **商家操作指南** 回答用户（商家）在后台操作时遇到的问题。\n"
            f"回答应尽可能清晰、步骤化。如果信息不足，请说明根据现有指南无法回答该操作问题。\n\n"
        )
        # 只获取 '商家操作' 类别的规则
        seller_rules = get_rules_by_category('商家操作') # [source: 222] 调用新函数
        context += "--- 商家操作指南 ---\n"
        if seller_rules:
            for rule in seller_rules:
                q = rule.get('question', 'N/A')
                a = rule.get('answer', 'N/A')
                context += f"Q: {q}\nA: {a}\n\n"
        else:
            context += "（抱歉，暂无相关的商家操作指南信息）\n" # [source: 223]
        context += "--- 指南结束 ---\n\n"
        # 注意：在这个场景下，我们只给 AI 提供商家相关的规则

    # === 场景 1：商品详情 (仅当不是商家帮助场景时才检查) ===
    elif product_id:
        print(f"--- [qa.py] 处理商品详情场景 (ID: {product_id}) ---")
        try:
            product = Product.query.get(int(product_id)) # [source: 224]
            if product:
                is_product_context = True
                print(f"--- [qa.py] 成功找到商品: {product.name} ---")
                # 构建商品信息字符串 (与之前步骤相同)
                product_context_str = f"--- 关于商品 '{product.name}' (ID: {product.id}) 的信息 ---\n"
                product_context_str += f"描述: {product.description or '暂无描述'}\n"
                product_context_str += f"价格: {product.price:.2f} 元\n"
                product_context_str += f"当前库存: {product.stock} 件\n"
                product_context_str += f"由卖家 '{product.seller.username}' 提供\n"
                product_context_str += "--- 商品信息结束 ---\n\n" # [source: 225]

                # 设置商品场景的指令
                prompt_instruction = (
                    f"用户正在查看商品 '{product.name}'。请主要根据以下提供的 **特定商品信息** 来回答用户的问题。\n"
                    f"如果商品信息不足以回答，可以参考下面补充的通用平台规则。\n"
                    f"如果根据所有信息都无法回答，请说明“根据现有信息无法回答关于此商品的问题”。\n\n"
                ) # [source: 226]

                context += product_context_str # 添加商品信息到上下文

                # 在商品场景下，加载通用规则作为补充信息
                all_rules = get_all_rules()
                general_context = "--- (补充) 平台规则与常见问题解答 ---\n" # 标题稍作修改
                if all_rules:
                    for rule in all_rules:
                        # (可选) 可以在这里过滤掉商品信息中已包含的内容，避免重复
                        # if rule.get('category') != '购物流程': # 简单示例
                        q = rule.get('question', 'N/A')
                        a = rule.get('answer', 'N/A')
                        general_context += f"Q: {q}\nA: {a}\n\n"
                else:
                    general_context += "（抱歉，目前没有可参考的通用规则信息）\n"
                general_context += "--- 平台信息结束 ---\n\n"
                context += general_context # 追加通用规则 # [source: 227]

            else:
                print(f"!!! [qa.py] 警告：数据库中未找到 ID 为 {product_id} 的商品 !!!") # [source: 228]
                # 未找到商品，将按通用场景处理 (因为 is_product_context 仍为 False)

        except (ValueError, TypeError) as e:
            print(f"!!! [qa.py] 错误：接收到的 product_id ({product_id}) 格式无效: {e} !!!")
        except Exception as e:
            print(f"!!! [qa.py] 错误：查询商品 (ID: {product_id}) 时出错: {e} !!!")

    # === 场景 2：通用提问 (如果既不是商家帮助，也没成功处理商品上下文) ===
    # is_product_context 仅在商品场景成功时为 True
    if not context_type and not is_product_context: # [source: 229] 确认是通用场景
        print("--- [qa.py] 处理通用规则场景 ---")
        prompt_instruction = (
            f"请仔细阅读以下关于我们平台的信息，并仅根据这些信息回答用户的问题。\n"
            f"如果信息不足以回答，请直接说明“根据我所掌握的信息，无法回答这个问题”。\n\n"
        )
        # 只加载并添加通用规则
        all_rules = get_all_rules()
        general_context = "--- 平台规则与常见问题解答 ---\n"
        if all_rules:
            for rule in all_rules:
                q = rule.get('question', 'N/A')
                a = rule.get('answer', 'N/A')
                general_context += f"Q: {q}\nA: {a}\n\n"
        else:
            general_context += "（抱歉，目前没有可参考的规则信息）\n"
        general_context += "--- 平台信息结束 ---\n\n"
        context += general_context # [source: 230]

    # --- 最后检查，确保有指令和上下文 ---
    if not prompt_instruction or not context:
        print("!!! [qa.py] 警告：未能确定场景或上下文为空，使用默认 Prompt !!!")
        # 提供一个非常基础的后备指令
        prompt_instruction = "请根据你的知识回答以下问题：\n"
        context = "(无特定上下文信息)\n"


    # --- 构建最终 Prompt ---
    final_prompt = (
        f"{prompt_instruction}" # 使用根据场景确定的指令
        f"{context}"          # 使用根据场景组合的上下文
        f"用户的问题是：\n{user_question}\n\n"
        f"请回答："
    ) # [source: 231]

    print(f"--- [qa.py] 构建的最终 Prompt (前 400 字符): {final_prompt[:400]}... ---")

    # --- 调用 AI 服务并返回结果 ---
    answer = get_openai_response(final_prompt)
    print(f"--- [qa.py] 返回的答案 (前 100 字符): {answer[:100]}... ---")
    return jsonify({"answer": answer})