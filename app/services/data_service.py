# app/services/data_service.py
import logging # 引入日志模块
from ..models import FaqRule # 导入 FaqRule 模型
from .. import db # 导入数据库实例 db 用于查询

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 全局变量，用于缓存从数据库加载的数据 ---
# 缓存结构将保持为 list of dicts，以兼容 qa.py 的现有逻辑
_rules_data_cache = None

def load_rules_from_db():
    """
    从数据库的 FaqRule 表加载所有规则/FAQ 数据。
    将 SQLAlchemy 对象转换为字典列表。
    """
    logger.info("Attempting to load rules from database...")
    try:
        # 查询数据库，获取所有 FaqRule 记录，按 ID 排序
        rules_objects = FaqRule.query.order_by(FaqRule.id).all()

        # 将查询结果 (SQLAlchemy 对象列表) 转换为字典列表
        # 这是为了保持与之前 JSON 结构一致，方便 qa.py 使用
        rules_list_of_dicts = [
            {
                'id': rule.id,
                'category': rule.category,
                'question': rule.question,
                'answer': rule.answer
            }
            for rule in rules_objects
        ]
        logger.info(f"Successfully loaded {len(rules_list_of_dicts)} rules from database.")
        return rules_list_of_dicts
    except Exception as e:
        # 处理可能的数据库查询错误
        logger.error(f"Error loading rules from database: {e}", exc_info=True) # 记录详细错误信息
        return [] # 出错时返回空列表

def get_all_rules():
    """
    获取所有规则/FAQ 数据（优先从缓存读取）。
    这是供其他模块调用的主要接口。
    """
    global _rules_data_cache
    if _rules_data_cache is None:
        logger.info("Rules cache is empty. Loading from database...")
        # 缓存为空，调用数据库加载函数
        _rules_data_cache = load_rules_from_db()
    else:
        logger.info("Using cached rules data.")
    # 返回缓存数据（如果加载失败，load_rules_from_db 会返回空列表）
    return _rules_data_cache if _rules_data_cache is not None else []

def clear_rules_cache():
    """
    清除规则数据的内存缓存。
    (这个函数将在后面管理员修改规则时被调用)
    """
    global _rules_data_cache
    logger.warning("Clearing rules data cache.")
    _rules_data_cache = None

def get_rules_by_category(category):
    """
    根据类别筛选规则/FAQ (从缓存或数据库获取数据)。
    """
    all_rules_dicts = get_all_rules() # 获取所有规则 (此函数会处理缓存和数据库加载)

    if not category:
        logger.warning("Category not provided for filtering.")
        return []

    category_lower = category.lower()
    logger.info(f"Filtering rules for category: '{category_lower}'")

    # 在获取到的字典列表上进行筛选 (逻辑不变)
    filtered_rules = [
        rule for rule in all_rules_dicts
        if rule.get('category', '').lower() == category_lower
    ]
    logger.info(f"Found {len(filtered_rules)} rules for category '{category_lower}'.")
    return filtered_rules

def find_rules_by_keyword(keyword):
    """
    根据关键词在问题或答案中查找规则 (从缓存或数据库获取数据)。
    """
    rules_dicts = get_all_rules() # 获取所有规则 (此函数会处理缓存和数据库加载)

    if not keyword:
        logger.warning("Keyword not provided for searching.")
        return []

    keyword_lower = keyword.lower()
    logger.info(f"Searching rules with keyword: '{keyword_lower}'")

    # 在获取到的字典列表上进行搜索 (逻辑不变)
    found_rules = [
        rule for rule in rules_dicts
        if keyword_lower in rule.get('question', '').lower() or \
           keyword_lower in rule.get('answer', '').lower()
    ]
    logger.info(f"Found {len(found_rules)} rules matching keyword '{keyword_lower}'.")
    return found_rules

# --- 不再需要在模块加载时预加载，改为首次调用 get_all_rules 时加载 ---
logger.info("[data_service] Module initialized. Rules will be loaded from DB on first request.")