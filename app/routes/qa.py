# app/routes/qa.py
from flask import Blueprint, request, jsonify
# 从 app/services 文件夹导入我们刚写的 AI 对话函数
from ..services.openai_service import get_openai_response

# 创建一个名叫 'qa' 的蓝图
qa = Blueprint('qa', __name__)

@qa.route('/ask', methods=['POST']) # 只接受 POST 请求
def ask_question():
    """接收前端发来的问题，调用 OpenAI 服务并返回答案"""
    print("--- 收到 /ask 请求 ---") # 调试信息

    # 检查请求过来的数据是不是 JSON 格式
    if not request.is_json:
        print("!!! 请求不是 JSON 格式 !!!")
        return jsonify({"error": "请求必须是 JSON 格式"}), 400 # 返回错误信息和状态码 400

    # 从 JSON 数据中获取 'question' 字段
    data = request.get_json()
    question = data.get('question')
    print(f"--- 收到的问题: {question} ---")

    # 简单验证问题是否存在
    if not question:
        print("!!! 请求中缺少 'question' 字段 !!!")
        return jsonify({"error": "请求中缺少 'question' 字段"}), 400

    # 调用我们的 AI 服务函数获取回答
    answer = get_openai_response(question)

    # 将回答包装成 JSON 格式返回给前端
    print(f"--- 返回的答案: {answer[:100]}... ---") # 打印部分答案用于调试
    return jsonify({"answer": answer})