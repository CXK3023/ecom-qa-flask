# filename: generate_products_grok_structured_v2.py
import os
import csv
import logging
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm

# --- 配置区域 ---
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY", "YOUR_XAI_API_KEY_HERE")
XAI_API_BASE = os.getenv("XAI_API_BASE", "YOUR_XAI_API_BASE_HERE")

MODEL_NAME = "grok-3-fast-latest"
TEMPERATURE = 1 # (警告：极高温度，建议测试 0.7-1.0)

TOTAL_ITEMS_PER_CATEGORY = 150
OUTPUT_FILENAME = "generated_products_grok_structured_v2.csv" # 更新文件名
FIXED_SELLER_USERNAME = "seller01@cxk.com"
MAX_WORKERS = 8
API_RETRY_DELAY = 5
MAX_RETRIES_PER_ITEM = 3

# === 品类详细信息 ===
CATEGORY_DETAILS = {
    "电子产品": {
        "brands": ["华为", "小米", "OPPO", "vivo", "Apple", "联想", "戴尔", "索尼", "三星", "小度", "天猫精灵", "Anker", "Bose", "大疆"],
        "product_lines": ["智能手机", "笔记本电脑", "平板电脑", "蓝牙耳机", "智能手表", "充电宝", "数据线", "显示器", "智能音箱", "无人机", "相机", "游戏主机配件"]
    },
    "服饰鞋包": {
        "brands": ["优衣库", "Zara", "H&M", "李宁", "安踏", "Nike", "Adidas", "森马", "太平鸟", "波司登", "COACH", "Michael Kors", "Tory Burch"],
        "product_lines": ["T恤", "衬衫", "连衣裙", "牛仔裤", "休闲裤", "羽绒服", "运动鞋", "休闲鞋", "高跟鞋", "双肩背包", "单肩包", "手提包", "钱包", "帽子", "围巾"]
    },
    "家居生活": {
        "brands": ["美的", "苏泊尔", "九阳", "小米米家", "网易严选", "水星家纺", "富安娜", "洁丽雅", "茶花", "乐扣乐扣"],
        "product_lines": ["电饭煲", "炒锅", "四件套", "被子", "枕头", "毛巾", "浴巾", "收纳箱", "垃圾桶", "拖把", "保温杯", "马克杯", "餐具套装", "香薰"]
    },
    "图书音像": {
        "brands": ["中信出版社", "人民邮电出版社", "机械工业出版社", "电子工业出版社", "新经典文化", "磨铁图书", "索尼音乐", "环球音乐", "华纳音乐"],
        "product_lines": ["小说", "经管书籍", "计算机书籍", "教材", "考试用书", "儿童绘本", "人物传记", "流行音乐CD", "古典音乐黑胶", "电影DVD", "电视剧蓝光碟"]
    },
    "美妆个护": {
        "brands": ["欧莱雅", "兰蔻", "雅诗兰黛", "SK-II", "完美日记", "花西子", "百雀羚", "自然堂", "海飞丝", "潘婷", "舒肤佳", "高露洁", "香奈儿", "迪奥"],
        "product_lines": ["洗面奶", "爽肤水", "精华液", "面霜", "面膜", "口红", "眼影", "粉底液", "洗发水", "沐浴露", "牙膏", "香水", "防晒霜", "身体乳"]
    },
    "运动户外": {
        "brands": ["迪卡侬", "探路者", "骆驼", "北面(The North Face)", "哥伦比亚", "Keep", "始祖鸟", "佳明", "牧高笛"],
        "product_lines": ["冲锋衣", "速干衣", "登山鞋", "帐篷", "睡袋", "登山杖", "运动手环", "跑步机", "瑜伽垫", "哑铃", "户外背包", "自行车", "滑雪装备"]
    },
    "母婴用品": {
        "brands": ["帮宝适", "好奇", "Babycare", "好孩子", "巴拉巴拉", "乐高", "美赞臣", "爱他美", "飞鹤", "贝亲"],
        "product_lines": ["纸尿裤", "婴儿湿巾", "奶瓶", "奶粉", "婴儿推车", "安全座椅", "童装", "童鞋", "积木玩具", "早教机", "辅食", "孕妇装"]
    },
}
# --- 品类详细信息结束 ---

# --- (日志配置) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- (xAI 客户端初始化) ---
client = None
if XAI_API_KEY == "YOUR_XAI_API_KEY_HERE" or XAI_API_BASE == "YOUR_XAI_API_BASE_HERE":
    logger.warning("请在脚本中或 .env 文件中设置您的 XAI_API_KEY 和 XAI_API_BASE")
else:
    try:
        client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_API_BASE, timeout=60.0)
        logger.info(f"xAI/Grok 客户端初始化成功，将使用模型: {MODEL_NAME}，端点: {XAI_API_BASE}")
    except Exception as e:
        logger.error(f"创建 xAI/Grok 客户端失败: {e}", exc_info=True); exit()

# --- (CSV 写入锁) ---
csv_writer_lock = threading.Lock()

# --- (Prompt 生成函数) ---
def generate_single_item_prompt(category, brand, product_line):
    prompt = f"""
你是一位富有创意的中国电商平台商品信息录入员。
请为【{category}】品类，专注于生成一个与品牌“{brand}”和产品线“{product_line}”相关的商品数据。
请严格按照以下 CSV 格式输出，**仅输出这一行商品数据**，不要包含表头，字段间用英文逗号 `,` 分隔：
Name,Description,Price,Stock,Seller Username

内容要求 (请务必遵守):
1.  **Name (商品名称)**: 生成一个具体且吸引人的商品名称，**必须**体现出品牌“{brand}”和产品线“{product_line}”的特点。例如，可以是 `{brand} {product_line} 新升级款` 或 `{brand} 明星{product_line}` 等形式，并加入具体规格或型号。面向中国市场。
2.  **Description (商品描述)**: (重点！) 生成详细、丰富、吸引人的商品描述，**至少 60 字**，要与 `{brand}` 品牌和 `{product_line}` 产品线紧密相关。**避免通用或与其他商品重复**。描述应包含：
    * 围绕 `{brand}` 和 `{product_line}` 的核心功能/独特卖点。
    * 相关的主要规格参数 (材质、尺寸、颜色、容量、技术规格等，根据 `{category}` 和 `{product_line}` 选择)。
    * 结合 `{brand}` 和 `{product_line}` 的使用场景。
    * (可选) 提及 `{brand}` 的品牌理念（如果了解或可以合理虚构）。
    * 使用 `\\n` 作为换行符来分段。
3.  **Price (价格)**: 生成符合 `{brand}` 品牌定位和 `{product_line}` 产品在中国市场价位的、看起来真实的人民币价格（浮点数）。
4.  **Stock (库存)**: 生成 50 到 5000 之间的随机整数库存量。
5.  **Seller Username (卖家用户名)**: 固定填写 `{FIXED_SELLER_USERNAME}`。

请确保生成的 Name 和 Description **高度相关**于指定的 `{brand}` 和 `{product_line}`，且内容新颖独特。
"""
    return prompt

# --- (API 调用函数) ---
def call_api_single(prompt, model_name, category, item_index):
    if not client: logger.error(f"[{category}-{item_index}] 客户端未初始化"); return None
    logger.debug(f"[{category}-{item_index}] 调用模型 {model_name} (Temp: {TEMPERATURE})...")
    try:
        completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=TEMPERATURE, max_tokens=1000)
        if completion.choices and completion.choices[0].message: return completion.choices[0].message.content.strip()
        else: logger.error(f"[{category}-{item_index}] API 响应格式不符合预期: {completion}"); return None
    except RateLimitError: logger.warning(f"[{category}-{item_index}] 速率限制，等待 {API_RETRY_DELAY} 秒后重试..."); time.sleep(API_RETRY_DELAY); return "RATE_LIMIT_RETRY"
    except APIConnectionError as e: logger.error(f"[{category}-{item_index}] API 连接错误: {e}"); return None
    except APIError as e:
        logger.error(f"[{category}-{item_index}] API 返回错误: {e}")
        if hasattr(e, 'status_code') and e.status_code == 429: logger.warning(f"[{category}-{item_index}] 速率限制 (429)，等待 {API_RETRY_DELAY} 秒后重试..."); time.sleep(API_RETRY_DELAY); return "RATE_LIMIT_RETRY"
        if "timeout" in str(e).lower(): logger.warning(f"[{category}-{item_index}] 超时，等待 {API_RETRY_DELAY} 秒后重试..."); time.sleep(API_RETRY_DELAY); return "RATE_LIMIT_RETRY"
        return None
    except Exception as e: logger.error(f"[{category}-{item_index}] 调用 API 时未知错误: {e}", exc_info=True); return None

# --- (CSV 解析函数) ---
def parse_single_item_csv(response_text, category, item_index):
    if not response_text: return None
    fields = response_text.split(',')
    expected_fields = 5
    if len(fields) >= expected_fields:
        name = fields[0].strip()
        desc = ','.join(fields[1:-3]).strip().replace('\\n','\n')
        price_str = fields[-3].strip()
        stock_str = fields[-2].strip()
        seller = fields[-1].strip()
        try: price = float(price_str); price = abs(price) if price < 0 else price
        except ValueError: logger.warning(f"[{category}-{item_index}] 价格格式错误: '{price_str}'"); return None
        try:
            stock = int(stock_str)
            if stock < 50: stock = random.randint(50, 500)
            if stock > 5000: stock = random.randint(3000, 5000)
        except ValueError: logger.warning(f"[{category}-{item_index}] 库存格式错误: '{stock_str}'"); return None
        if seller != FIXED_SELLER_USERNAME: logger.warning(f"[{category}-{item_index}] 卖家用户名错误: '{seller}'，已修正"); seller = FIXED_SELLER_USERNAME
        if name.startswith('"') and name.endswith('"'): name = name[1:-1]
        if desc.startswith('"') and desc.endswith('"'): desc = desc[1:-1]
        desc = desc.replace('""', '"')
        if name and desc: return [name, desc, f"{price:.2f}", str(stock), seller]
        else: logger.warning(f"[{category}-{item_index}] 解析后名称或描述为空"); return None
    else: logger.warning(f"[{category}-{item_index}] 解析响应格式错误 (字段数: {len(fields)}): {response_text[:100]}..."); return None

# --- (任务执行函数) ---
def generate_and_write_item(category, brand, product_line, item_index, writer, pbar):
    retries = MAX_RETRIES_PER_ITEM
    task_id = f"{category}-{brand}-{product_line}-{item_index}"
    while retries > 0:
        prompt = generate_single_item_prompt(category, brand, product_line)
        response_text = call_api_single(prompt, MODEL_NAME, category, item_index)
        if response_text == "RATE_LIMIT_RETRY": continue
        elif response_text:
            parsed_row = parse_single_item_csv(response_text, category, item_index)
            if parsed_row:
                with csv_writer_lock: writer.writerow(parsed_row)
                pbar.update(1); return True
            else: logger.warning(f"[{task_id}] 解析失败，重试 ({MAX_RETRIES_PER_ITEM - retries + 1}/{MAX_RETRIES_PER_ITEM})"); retries -= 1; time.sleep(2)
        else: logger.error(f"[{task_id}] API 调用失败或返回空，重试 ({MAX_RETRIES_PER_ITEM - retries + 1}/{MAX_RETRIES_PER_ITEM})"); retries -= 1; time.sleep(API_RETRY_DELAY)
    logger.error(f"!!! [{task_id}] 达到最大重试次数，生成失败 !!!"); return False

# --- (主函数 - 修正语法错误) ---
def main():
    categories_available = list(CATEGORY_DETAILS.keys())
    total_required = len(categories_available) * TOTAL_ITEMS_PER_CATEGORY
    logger.info(f"将为 {len(categories_available)} 个品类，每个生成 {TOTAL_ITEMS_PER_CATEGORY} 条数据。")
    logger.info(f"总共需要生成 {total_required} 条商品数据 (模型: {MODEL_NAME}, Temp: {TEMPERATURE})，使用 {MAX_WORKERS} 个并发线程。")
    tasks = []
    item_counter = 0
    for category in categories_available:
        details = CATEGORY_DETAILS[category]
        brands = details.get("brands", ["通用品牌"])
        product_lines = details.get("product_lines", ["通用产品"])

        # === vvv 修正这里的语法 vvv ===
        if not brands:
            brands = ["通用品牌"]
        if not product_lines:
            product_lines = ["通用产品"]
        # === ^^^ 修正结束 ^^^ ===

        for _ in range(TOTAL_ITEMS_PER_CATEGORY):
            item_counter += 1
            selected_brand = random.choice(brands)
            selected_product_line = random.choice(product_lines)
            tasks.append((category, selected_brand, selected_product_line, item_counter))

    write_header = not os.path.exists(OUTPUT_FILENAME) or os.path.getsize(OUTPUT_FILENAME) == 0
    successful_count = 0; failed_count = 0
    try:
        with open(OUTPUT_FILENAME, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if write_header: writer.writerow(['Name', 'Description', 'Price', 'Stock', 'Seller Username']); logger.info(f"已写入 CSV 表头到 {OUTPUT_FILENAME}")
            with tqdm(total=total_required, desc="Generating Products", unit="item") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_task = {executor.submit(generate_and_write_item, task[0], task[1], task[2], task[3], writer, pbar): task for task in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            success = future.result()
                            if success: successful_count += 1
                            else: failed_count += 1
                        except Exception as exc: failed_count += 1; logger.error(f"Task {task[0]}-{task[1]}-{task[2]}-{task[3]} generated an exception: {exc}", exc_info=True); pbar.update(1)
    except IOError as e: logger.error(f"无法写入文件 {OUTPUT_FILENAME}: {e}", exc_info=True); return
    logger.info(f"\n=== 数据生成完成 ==="); logger.info(f"成功生成并写入 {successful_count} / {total_required} 条商品数据。")
    if failed_count > 0: logger.warning(f"有 {failed_count} 条商品数据生成失败。")
    logger.info(f"数据已保存到文件: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    if client: main()
    else: logger.error("客户端未成功初始化，无法运行数据生成。请检查 API Key 和 Base URL 配置。")