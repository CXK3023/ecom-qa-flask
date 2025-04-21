# app/services/data_service.py
import json  # Python 内置的处理 JSON 的库
import os    # 用来处理文件路径

# --- 全局变量，用于缓存加载的数据 ---
# 我们把读取到的规则存放在这里，避免每次需要都去读文件
_rules_data = None
_rules_file_path = None # 缓存文件路径

def _get_rules_file_path():
    """
    动态计算 rules.json 文件的绝对路径。
    这样无论我们的脚本在哪里运行，都能找到它。
    """
    global _rules_file_path
    if _rules_file_path is None:
        # __file__ 是当前 data_service.py 文件的路径
        # os.path.abspath(__file__) 获取它的绝对路径
        # os.path.dirname(...) 获取它所在的目录 (app/services)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 我们知道 rules.json 在项目根目录的 data 文件夹下
        # 从 app/services 往上走两级 ('..', '..') 就到了项目根目录
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        # 然后拼接上 'data' 和 'rules.json' 得到最终路径
        _rules_file_path = os.path.join(project_root, 'data', 'rules.json')
        print(f"--- [data_service] 规则文件路径确定为: {_rules_file_path} ---") # 打印路径方便调试
    return _rules_file_path

def load_rules_from_json():
    """
    从 rules.json 文件加载规则/FAQ 数据。
    使用缓存避免重复读取文件。
    """
    global _rules_data
    # 如果 _rules_data 已经有内容了（说明之前加载过），就直接返回它
    if _rules_data is not None:
        print("--- [data_service] 使用已缓存的规则数据 ---")
        return _rules_data

    print("--- [data_service] 首次加载规则数据或缓存为空 ---")
    file_path = _get_rules_file_path() # 获取文件路径

    try:
        # 'r' 表示读取模式, encoding='utf-8' 确保能正确处理中文等字符
        with open(file_path, 'r', encoding='utf-8') as f:
            # json.load(f) 会读取文件内容并把它解析成 Python 的数据结构 (这里是列表)
            loaded_data = json.load(f)
            _rules_data = loaded_data  # 将加载的数据存入缓存变量
            print(f"--- [data_service] 成功加载 {len(_rules_data)} 条规则 ---")
            return _rules_data
    except FileNotFoundError:
        # 如果文件不存在
        print(f"!!! [data_service] 错误：规则文件未找到: {file_path} !!!")
        _rules_data = [] # 文件找不到，返回一个空列表，避免程序出错
        return _rules_data
    except json.JSONDecodeError as e:
        # 如果文件内容不是有效的 JSON 格式
        print(f"!!! [data_service] 错误：解析 rules.json 文件失败: {e} !!!")
        print("--- 请仔细检查 rules.json 文件的格式是否正确（逗号、引号等）！ ---")
        _rules_data = [] # 解析失败，也返回空列表
        return _rules_data
    except Exception as e:
        # 捕获其他可能发生的意外错误
        print(f"!!! [data_service] 加载规则文件时发生未知错误: {e} !!!")
        _rules_data = [] # 发生未知错误，返回空列表
        return _rules_data

def get_all_rules():
    """
    获取所有已加载的规则/FAQ。
    这个函数是给其他模块调用的接口。
    """
    # 它会调用 load_rules_from_json()，该函数内部会处理缓存逻辑
    return load_rules_from_json()

# app/services/data_service.py (添加这个新函数)

def get_rules_by_category(category):
    """
    根据类别筛选规则/FAQ。
    Args:
        category (str): 需要筛选的类别名称 (例如 '商家操作', '售后')。
    Returns:
        list: 包含所有匹配类别规则的字典列表，如果找不到则返回空列表。
    """
    all_rules = get_all_rules() # 首先获取所有规则 (会使用缓存)

    if not category: # 如果传入的类别是空的，直接返回空列表
        return []

    category_lower = category.lower() # 将传入的类别名称转为小写，用于不区分大小写的比较

    # 使用列表推导式进行筛选
    filtered_rules = [
        rule for rule in all_rules # 遍历每一条规则
        # rule.get('category', '') 安全地获取规则的 'category' 字段，如果没有则返回空字符串
        # .lower() 将规则的类别也转为小写
        # 比较两者是否相等
        if rule.get('category', '').lower() == category_lower
    ]

    # 打印调试信息，显示筛选结果数量
    print(f"--- [data_service] 根据类别 '{category}' 筛选到 {len(filtered_rules)} 条规则 ---")
    return filtered_rules


# (可选) 简单的关键词查找功能 (我们明天可能会用到或改进它)
def find_rules_by_keyword(keyword):
    """
    根据关键词在问题或答案中查找规则 (忽略大小写)。
    """
    rules = get_all_rules() # 先获取所有规则
    if not keyword: # 如果关键词是空的，直接返回空列表
        return []
    keyword_lower = keyword.lower() # 将关键词转为小写，方便比较
    # 使用列表推导式筛选规则
    found_rules = [
        rule for rule in rules # 遍历每一条规则
        # 检查关键词是否在规则的 'question' 或 'answer' 字段中 (都转为小写比较)
        # rule.get('question', '') 如果没有 'question' 键，返回空字符串，避免错误
        if keyword_lower in rule.get('question', '').lower() or \
           keyword_lower in rule.get('answer', '').lower()
    ]
    return found_rules

# --- 模块首次被导入时，尝试预加载一次数据 ---
# 这样应用启动后，第一次需要规则数据时就可能直接从缓存读取了
print("--- [data_service] 模块初始化，尝试预加载规则数据 ---")
load_rules_from_json()