# ecom-qa-flask/data_generation_scripts/generate_evaluation_data.py
# Revised Version 4.4d (Debugging): Added detailed debug logging.
import os
import json
import csv
import logging
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm
import traceback
from io import StringIO # For robust CSV parsing

# --- Configuration Area ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Use environment variables with defaults
XAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_IF_NEEDED"))
XAI_API_BASE = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_API_BASE", "YOUR_DEFAULT_BASE_IF_NEEDED"))
MODEL_NAME = os.getenv("MODEL_NAME", "grok-3-fast") # Allow overriding model via env var

TEMPERATURE_QUESTION = 0.8
TEMPERATURE_PRODUCT = 0.9 # Keep slightly higher for product diversity

# --- Generation Quantity Config ---
NUM_QUESTIONS_PER_CATEGORY = 2 # <<-- 每个品类 *尝试* 生成的问题集数量 (会受限于唯一组合数)
NUM_RELATED_PRODUCTS_MIN = 3   # <<-- 每个问题生成的相关商品最小数量
NUM_RELATED_PRODUCTS_MAX = 5   # <<-- 每个问题生成的相关商品最大数量
NUM_DISTRACTOR_PRODUCTS = 50    # <<-- 每个问题生成的干扰商品数量 (来自不同组合)

# --- Output Files ---
# Using different filenames for debug version
OUTPUT_PRODUCT_CSV = "products_for_eval_v6_cn_v4.4_debug.csv"
OUTPUT_QUESTION_JSON = "test_questions_final_v6_cn_v4.4_debug.json"

# --- Other Config ---
FIXED_SELLER_USERNAME = "seller01@cxk.com"
MAX_WORKERS = 8 # Concurrency for product generation API calls
API_RETRY_DELAY = 5
MAX_RETRIES_PER_API_CALL = 3
PRODUCT_DESC_MIN_LENGTH = 40

# === 品类详细信息 (扩展版) ===
# (CATEGORY_DETAILS remains the same as provided in the original script)
CATEGORY_DETAILS = {
    "电子产品": {
        "brands": ["华为", "小米", "OPPO", "vivo", "Apple", "联想", "戴尔", "索尼", "三星", "小度", "天猫精灵", "Anker", "Bose", "大疆", "GoPro", "罗技"],
        "product_lines": ["智能手机", "笔记本电脑", "平板电脑", "蓝牙耳机", "智能手表", "充电宝", "数据线", "显示器", "智能音箱", "无人机", "相机", "游戏主机配件", "键盘", "鼠标", "路由器", "移动硬盘"]
    },
    "服饰鞋包": {
        "brands": ["优衣库", "Zara", "H&M", "李宁", "安踏", "Nike", "Adidas", "森马", "太平鸟", "波司登", "COACH", "Michael Kors", "Tory Burch", "CHARLES & KEITH", "热风"],
        "product_lines": ["T恤", "衬衫", "连衣裙", "牛仔裤", "休闲裤", "羽绒服", "夹克", "卫衣", "运动鞋", "休闲鞋", "高跟鞋", "靴子", "双肩背包", "单肩包", "手提包", "钱包", "帽子", "围巾", "皮带"]
    },
    "家居生活": {
        "brands": ["美的", "苏泊尔", "九阳", "小米米家", "网易严选", "水星家纺", "富安娜", "洁丽雅", "茶花", "乐扣乐扣", "无印良品(MUJI)", "名创优品"],
        "product_lines": ["四件套", "被子", "枕头", "毛巾", "浴巾", "收纳箱", "垃圾桶", "拖把", "香薰", "窗帘", "地毯", "摆件", "衣架", "纸巾", "清洁剂", "一次性用品"] # 调整了产品线，移除了厨具和部分小家电
    },
    "图书音像": {
        "brands": ["中信出版社", "人民邮电出版社", "机械工业出版社", "电子工业出版社", "新经典文化", "磨铁图书", "索尼音乐", "环球音乐", "华纳音乐", "后浪", "读客文化"],
        "product_lines": ["小说", "经管书籍", "计算机书籍", "教材", "考试用书", "儿童绘本", "人物传记", "漫画", "杂志", "流行音乐CD", "古典音乐黑胶", "电影DVD", "电视剧蓝光碟", "有声书"]
    },
    "美妆个护": {
        "brands": ["欧莱雅", "兰蔻", "雅诗兰黛", "SK-II", "完美日记", "花西子", "百雀羚", "自然堂", "海飞丝", "潘婷", "舒肤佳", "高露洁", "香奈儿", "迪奥", "资生堂", "薇诺娜", "云南白药牙膏"],
        "product_lines": ["洗面奶", "爽肤水", "精华液", "面霜", "面膜", "口红", "眼影", "粉底液", "卸妆水", "洗发水", "护发素", "沐浴露", "牙膏", "牙刷", "香水", "防晒霜", "身体乳", "剃须刀", "卫生巾"]
    },
    "运动户外": {
        "brands": ["迪卡侬", "探路者", "骆驼", "北面(The North Face)", "哥伦比亚", "Keep", "始祖鸟", "佳明(Garmin)", "牧高笛", "斐乐(FILA)", "彪马(PUMA)", "斯凯奇(Skechers)"],
        "product_lines": ["冲锋衣", "速干衣", "登山鞋", "帐篷", "睡袋", "登山杖", "运动手环", "跑步机", "瑜伽垫", "哑铃", "户外背包", "自行车", "滑雪装备", "泳衣", "篮球", "羽毛球拍"]
    },
    "母婴用品": {
        "brands": ["帮宝适", "好奇", "Babycare", "好孩子(Goodbaby)", "巴拉巴拉", "乐高(LEGO)", "美赞臣", "爱他美", "飞鹤", "贝亲(Pigeon)", "英氏(YeeHoO)", "全棉时代"],
        "product_lines": ["纸尿裤", "婴儿湿巾", "奶瓶", "奶粉", "婴儿推车", "安全座椅", "童装", "童鞋", "积木玩具", "早教机", "辅食", "孕妇装", "吸奶器", "婴儿床", "爬行垫"]
    },
    "食品生鲜": {
        "brands": ["三只松鼠", "良品铺子", "百草味", "蒙牛", "伊利", "康师傅", "统一", "农夫山泉", "洽洽", "卫龙", "奥利奥", "旺旺", "金龙鱼"],
        "product_lines": ["零食", "坚果炒货", "饼干蛋糕", "糖果巧克力", "方便食品", "牛奶乳品", "饮料饮品", "茗茶", "咖啡", "粮油调味", "生鲜水果", "蔬菜", "肉禽蛋", "海鲜水产"]
    },
    "家电": {
        "brands": ["海尔", "格力", "美的", "西门子", "博世", "小米米家", "戴森(Dyson)", "飞利浦(Philips)", "TCL", "海信", "苏泊尔", "九阳", "小熊电器"],
        "product_lines": ["冰箱", "洗衣机", "空调", "电视", "热水器", "油烟机", "燃气灶", "消毒柜", "洗碗机", "吸尘器", "扫地机器人", "空气净化器", "电风扇", "取暖器", "微波炉", "电烤箱", "电磁炉", "净水器"]
    },
    "家具": {
        "brands": ["宜家(IKEA)", "林氏木业", "全友家居", "顾家家居", "曲美家居", "源氏木语", "网易严选", "芝华仕(CHEERS)", "索菲亚"],
        "product_lines": ["床", "沙发", "衣柜", "餐桌", "椅子", "书桌", "书柜", "鞋柜", "床垫", "电视柜", "茶几", "梳妆台", "置物架", "灯具"]
    },
    "厨具": {
        "brands": ["苏泊尔", "九阳", "双立人(Zwilling)", "菲仕乐(Fissler)", "WMF", "爱仕达", "炊大皇", "康巴赫", "乐扣乐扣(Lock&Lock)", "特百惠(Tupperware)", "张小泉"],
        "product_lines": ["炒锅", "煎锅", "汤锅", "蒸锅", "压力锅", "刀具套装", "砧板", "碗盘碟", "筷子勺子", "保鲜盒", "烘焙模具", "厨房置物架", "锅铲", "电饭煲", "电水壶", "咖啡机", "榨汁机", "豆浆机"] # 补充了部分小厨电
    },
    "汽车用品": {
        "brands": ["米其林(Michelin)", "普利司通(Bridgestone)", "固特异(Goodyear)", "3M", "博世(Bosch)", "飞利浦(Philips)", "龟牌(Turtle Wax)", "倍思(Baseus)", "尤利特", "沿途", "车仆", "铁将军"],
        "product_lines": ["轮胎", "机油", "添加剂", "行车记录仪", "车载充电器", "手机支架", "汽车香水", "脚垫", "座套", "洗车液", "玻璃水", "应急启动电源", "车载冰箱", "导航仪", "胎压监测", "空气净化器"]
    },
    "宠物生活": {
        "brands": ["皇家(Royal Canin)", "宝路(Pedigree)", "伟嘉(Whiskas)", "耐威克", "渴望(Orijen)", "爱肯拿(Acana)", "Petkit小佩", "Catlink", "疯狂的小狗", "网易严选宠物", "麦富迪"],
        "product_lines": ["狗粮", "猫粮", "宠物零食", "猫砂", "狗窝", "猫窝", "宠物玩具", "宠物服装", "牵引绳", "项圈", "宠物清洁美容", "智能喂食器", "饮水器", "宠物尿垫", "水族箱", "鱼粮"]
    },
    "医药保健": {
        "brands": ["汤臣倍健", "Swisse", "GNC", "Move Free", "同仁堂", "云南白药", "999", "修正", "鱼跃(Yuwell)", "欧姆龙(OMRON)", "强生(Johnson & Johnson)", "博士伦(Bausch Lomb)"],
        "product_lines": ["维生素", "钙片", "蛋白粉", "鱼油", "益生菌", "OTC药品(感冒, 肠胃, 止痛)", "血压计", "血糖仪", "体温计", "雾化器", "制氧机", "隐形眼镜", "护理液", "创可贴", "医用口罩", "消毒液", "轮椅", "拐杖"]
    },
    "玩具乐器": {
        "brands": ["乐高(LEGO)", "孩之宝(Hasbro)", "费雪(Fisher-Price)", "奥迪双钻", "泡泡玛特(Pop Mart)", "万代(Bandai)", "美泰(Mattel)", "布鲁可", "雅马哈(Yamaha)", "卡西欧(Casio)", "珠江钢琴", "罗兰(Roland)"],
        "product_lines": ["积木", "遥控玩具", "毛绒玩具", "益智玩具", "手办模型", "盲盒", "芭比娃娃", "奥特曼", "儿童自行车", "滑板车", "电子琴", "吉他", "尤克里里", "钢琴", "口琴", "学习机"]
    },
    "珠宝钟表": {
        "brands": ["周大福", "周生生", "老凤祥", "六福珠宝", "潮宏基", "卡地亚(Cartier)", "蒂芙尼(Tiffany & Co.)", "宝格丽(BVLGARI)", "施华洛世奇(Swarovski)", "潘多拉(Pandora)", "浪琴(Longines)", "天梭(Tissot)", "卡西欧(Casio)", "西铁城(Citizen)", "精工(Seiko)", "劳力士(Rolex)"],
        "product_lines": ["黄金饰品", "铂金饰品", "K金饰品", "钻石戒指", "项链", "吊坠", "耳环", "手镯", "手链", "投资金条", "机械表", "石英表", "智能手表(高端)", "时尚手表", "珠宝箱"]
    },
    "办公用品": {
        "brands": ["得力(Deli)", "晨光(M&G)", "国誉(Kokuyo)", "齐心(Comix)", "百乐(Pilot)", "三菱(Uni)", "斑马(Zebra)", "辉柏嘉(Faber-Castell)", "惠普(HP)", "爱普生(Epson)", "佳能(Canon)", "金士顿(Kingston)", "西数(WD)", "希捷(Seagate)"],
        "product_lines": ["笔类(中性笔,圆珠笔,铅笔)", "笔记本", "记事本", "文件管理(文件夹,档案袋)", "打印纸", "复印纸", "订书机", "计算器", "胶带", "剪刀", "打印机", "复印机", "扫描仪", "投影仪", "硒鼓墨盒", "U盘", "移动硬盘", "键盘鼠标套装", "碎纸机", "白板"]
    },
     "酒水饮料": {
        "brands": ["茅台", "五粮液", "洋河", "泸州老窖", "汾酒", "青岛啤酒", "雪花啤酒", "百威", "哈尔滨啤酒", "张裕", "长城", "奔富(Penfolds)", "拉菲(Lafite)", "可口可乐", "百事可乐", "农夫山泉", "怡宝", "统一", "康师傅", "元气森林", "巴黎水(Perrier)", "星巴克(Starbucks)"],
        "product_lines": ["白酒", "啤酒", "葡萄酒(红/白/桃红)", "洋酒(威士忌/白兰地)", "黄酒", "预调酒", "碳酸饮料", "果汁", "茶饮料", "咖啡饮料", "饮用水(矿泉水/纯净水)", "功能饮料", "植物蛋白饮料", "苏打水"]
    }
}
# --- 品类详细信息结束 ---

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)
# <<< 修改开始: 设置日志级别为 DEBUG >>>
logger.setLevel(logging.DEBUG)
# <<< 修改结束 >>>

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Optional: File Handler
# file_handler = logging.FileHandler('data_generation_cn_debug.log', encoding='utf-8') # Log to a separate file
# file_handler.setFormatter(log_formatter)
# logger.addHandler(file_handler)


# --- Global Variables and Locks ---
client = None
all_products_list = []
test_questions_list = []
product_id_counter = 0
product_list_lock = threading.Lock()
question_list_lock = threading.Lock()
id_counter_lock = threading.Lock()
csv_writer_lock = threading.Lock()

# --- Helper Functions ---
def initialize_client():
    global client
    if not XAI_API_KEY or XAI_API_KEY in ["YOUR_XAI_API_KEY_HERE", "YOUR_DEFAULT_KEY_IF_NEEDED", ""]:
        logger.error("XAI_API_KEY 未设置或包含占位符。请在 .env 文件中设置有效的 API 密钥。")
        return False
    if not XAI_API_BASE or XAI_API_BASE in ["YOUR_XAI_API_BASE_HERE", "YOUR_DEFAULT_BASE_IF_NEEDED", ""]:
        logger.error("XAI_API_BASE 未设置或包含占位符。请在 .env 文件中设置有效的 API 端点。")
        return False
    try:
        client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_API_BASE, timeout=90.0)
        logger.info(f"API 客户端初始化成功，端点: {XAI_API_BASE}, 模型: {MODEL_NAME}")
        return True
    except Exception as e:
        logger.error(f"创建或测试 API 客户端失败: {e}", exc_info=True)
        client = None
        return False

def get_unique_product_id():
    global product_id_counter
    with id_counter_lock:
        product_id_counter += 1
        return product_id_counter

def call_api(prompt, model, temperature, task_id, max_tokens=1000):
    if not client:
        logger.error(f"[{task_id}] API 客户端未初始化，无法调用。")
        return None
    retries = MAX_RETRIES_PER_API_CALL
    current_delay = API_RETRY_DELAY
    while retries > 0:
        try:
            # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] 尝试调用 client.chat.completions.create... Prompt(Start): {prompt[:150]}...")
            # <<< 添加日志结束 >>>
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] client.chat.completions.create 调用完成。")
            # <<< 添加日志结束 >>>
            raw_response = completion.choices[0].message.content.strip() if completion.choices and completion.choices[0].message else None
            if raw_response:
                 logger.debug(f"[{task_id}] Received non-empty response: {raw_response[:150]}...")
                 return raw_response
            else:
                 logger.warning(f"[{task_id}] API 调用成功但返回了空响应。")
                 return None

        except RateLimitError as e:
            # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] 捕获到 RateLimitError: {e}")
            # <<< 添加日志结束 >>>
            logger.warning(f"[{task_id}] 遇到速率限制: {e}。等待 {current_delay:.1f}s 重试 ({MAX_RETRIES_PER_API_CALL - retries + 1}/{MAX_RETRIES_PER_API_CALL})")
        except APIConnectionError as e:
             # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] 捕获到 APIConnectionError: {e}")
            # <<< 添加日志结束 >>>
            logger.warning(f"[{task_id}] API 连接错误: {e}。等待 {current_delay:.1f}s 重试 ({MAX_RETRIES_PER_API_CALL - retries + 1}/{MAX_RETRIES_PER_API_CALL})")
        except APIError as e:
            # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] 捕获到 APIError: {e}")
            # <<< 添加日志结束 >>>
            logger.error(f"[{task_id}] API 返回错误: {e}，状态码: {getattr(e, 'status_code', 'N/A')}, 类型: {getattr(e, 'type', 'N/A')}")
            if hasattr(e, 'status_code') and e.status_code == 429:
                logger.warning(f"[{task_id}] 速率限制 (429)，等待 {current_delay:.1f}s 重试...")
            elif "timeout" in str(e).lower():
                 logger.warning(f"[{task_id}] 超时，等待 {current_delay:.1f}s 重试...")
            else:
                 logger.warning(f"[{task_id}] 遇到可重试的 API 错误，等待 {current_delay:.1f}s 重试...")
        except Exception as e:
            # <<< 添加日志 >>>
            logger.debug(f"[{task_id}] 捕获到未知 Exception: {e}")
            # <<< 添加日志结束 >>>
            logger.error(f"[{task_id}] 调用 API 时未知错误: {e}", exc_info=True)
            logger.warning(f"[{task_id}] 遇到未知错误，尝试等待 {current_delay:.1f}s 重试...")

        # <<< 添加日志 >>>
        logger.debug(f"[{task_id}] 将等待 {current_delay:.1f} 秒后重试 (剩余 {retries-1} 次)...")
        # <<< 添加日志结束 >>>
        time.sleep(current_delay)
        retries -= 1
        current_delay *= 1.5 # Exponential backoff

    logger.error(f"[{task_id}] API 调用失败，已达到最大重试次数 {MAX_RETRIES_PER_API_CALL}。")
    return None

def parse_product_csv_line(line, task_id):
    # (No changes needed here)
    if not line:
        logger.warning(f"[{task_id}] 接收到空的 CSV 行进行解析。")
        return None
    try:
        clean_line = line.strip()
        f = StringIO(clean_line)
        reader = csv.reader(f, delimiter=',', quotechar='"', skipinitialspace=True)
        fields = next(reader)

        if len(fields) == 5:
            name = fields[0].strip()
            desc = fields[1].strip().replace('\\n', '\n')
            price_str = fields[2].strip()
            stock_str = fields[3].strip()
            seller = fields[4].strip()

            if not name or not desc:
                logger.warning(f"[{task_id}] 解析失败：名称或描述为空。Name: '{name}', Desc: '{desc[:50]}...' Line: {line[:80]}...")
                return None
            if len(desc) < PRODUCT_DESC_MIN_LENGTH:
                logger.warning(f"[{task_id}] 解析失败：描述太短 ({len(desc)} < {PRODUCT_DESC_MIN_LENGTH} chars)。 Name: {name} Line: {line[:80]}...")
                return None
            try:
                price = abs(float(price_str))
                if price <= 0:
                     logger.warning(f"[{task_id}] 解析警告：价格为零或负数 ({price_str})。 Name: {name}")
            except ValueError:
                logger.warning(f"[{task_id}] 解析失败：价格格式错误 '{price_str}'。 Name: {name} Line: {line[:80]}...")
                return None
            try:
                stock = max(50, min(int(stock_str), 5000))
            except ValueError:
                logger.warning(f"[{task_id}] 解析失败：库存格式错误 '{stock_str}'。 Name: {name} Line: {line[:80]}...")
                return None
            if seller != FIXED_SELLER_USERNAME:
                logger.warning(f"[{task_id}] 解析发现卖家错误 '{seller}'，已修正为 '{FIXED_SELLER_USERNAME}'。 Name: {name}")
                seller = FIXED_SELLER_USERNAME

            return [name, desc, f"{price:.2f}", str(stock), seller]
        else:
            logger.warning(f"[{task_id}] 解析失败：字段数量 ({len(fields)}) 不为 5。 Fields: {fields}");
            logger.debug(f"[{task_id}] Problematic raw line for parsing: '{line}'")
            return None
    except StopIteration:
        logger.error(f"[{task_id}] CSV 解析错误: 无法从处理后的行解析字段。 Cleaned Line: '{clean_line}' Original Line: '{line[:150]}...'")
        return None
    except csv.Error as e:
        logger.error(f"[{task_id}] CSV 解析库错误: {e}。 Line: '{line[:150]}...'")
        return None
    except Exception as e:
        logger.error(f"[{task_id}] 解析 CSV 行时发生意外错误: {e}。 Line: '{line[:150]}...'", exc_info=True)
        return None

def write_product_to_csv(product_data, writer):
    # (No changes needed here)
    with csv_writer_lock:
        try:
            writer.writerow(product_data)
        except Exception as e:
            logger.error(f"写入 CSV 时出错: {e}. Data: {product_data}", exc_info=True)


# --- Prompt Generation Function ---
def generate_prompt(prompt_type, **kwargs):
    # (No changes needed here compared to v4.4)
    if prompt_type == "generate_question":
        category = kwargs.get("category"); brand = kwargs.get("brand"); product_line = kwargs.get("product_line")
        return f"""为电商场景生成1个关于“{category}”品类下“{brand}”品牌“{product_line}”产品的中文购物咨询问题(15-25字)。直接输出问题。"""

    elif prompt_type == "generate_related":
        question = kwargs.get("question"); category = kwargs.get("category"); brand = kwargs.get("brand"); product_line = kwargs.get("product_line")
        previous_product_name = kwargs.get("previous_product_name")
        diversity_instruction = ""
        if previous_product_name:
            diversity_instruction = f" 请生成一个与商品 “{previous_product_name}” 【不同】的具体商品，可以是不同的型号、规格或特性。"
        else:
            diversity_instruction = ""

        return f"""用户问: "{question}" (品类: {category}, 品牌: {brand}, 产品线: {product_line})。
生成1个【高度相关】的具体商品数据，该商品应为品牌 "{brand}" 下产品线 "{product_line}" 中 **在中国市场常见的产品**。{diversity_instruction} # <<< Diversity instruction included here
严格按以下单行CSV格式输出 (字段必须用英文逗号`,`分隔, 文本字段必须用英文双引号`"`包裹):
"Name","Description","Price","Stock","{FIXED_SELLER_USERNAME}"
要求: 商品Name和Description应**符合中国市场习惯 (例如使用中文或常见中英结合命名)**。Description至少{PRODUCT_DESC_MIN_LENGTH}字, 内容独特且面向中国消费者, 可用 \\n 换行。Price是合理的人民币数字。Stock是50-5000整数。"""

    elif prompt_type == "generate_specific_product":
        category = kwargs.get("category")
        brand = kwargs.get("brand")
        product_line = kwargs.get("product_line")
        original_question = kwargs.get("original_question", "N/A")
        return f"""
为【{category}】品类，生成1个具体的商品数据: 品牌 "{brand}"，产品线 "{product_line}"。 **请确保这是一个在中国市场常见或熟知的产品型号**。
此商品将用作与问题 "{original_question}" 不直接相关的干扰项，但商品本身信息需合理且符合中国市场背景。

**必须严格遵循以下单行CSV格式输出，无表头，字段用`,`分隔，所有文本字段(Name, Description)用英文双引号`"`包裹:**
"Name","Description","Price","Stock","{FIXED_SELLER_USERNAME}"

内容要求:
1. Name: 体现 "{brand}" 和 "{product_line}"，包含具体型号/规格，**使用中文或中英结合命名 (符合中国市场)**。
2. Description: 详细描述 (至少{PRODUCT_DESC_MIN_LENGTH}字)，**面向中国消费者**。可用 \\n 换行。
3. Price: 合理的**人民币**价格 (数字)。
4. Stock: 50-5000 之间的整数。
5. Seller Username: 固定为 `{FIXED_SELLER_USERNAME}`。
"""
    else:
        logger.error(f"尝试生成未知的 prompt 类型: {prompt_type}")
        raise ValueError(f"未知的 prompt 类型: {prompt_type}")

# --- Core Task Processing Function ---
def process_category_question_set(category, details, csv_writer, pbar, selected_brand, selected_line, task_id_suffix):
    global all_products_list, test_questions_list
    task_id_prefix = task_id_suffix
    logger.info(f"[{task_id_prefix}] 开始处理组合: {category} / {selected_brand} / {selected_line}") # Existing INFO log

    brands = details.get("brands", [])
    product_lines = details.get("product_lines", [])
    if not brands: brands = ["通用品牌"]
    if not product_lines: product_lines = ["通用产品"]

    # <<< 添加日志 >>>
    logger.debug(f"[{task_id_prefix}] 准备生成问题 Prompt...")
    # <<< 添加日志结束 >>>
    question_prompt = generate_prompt("generate_question", category=category, brand=selected_brand, product_line=selected_line)
    # <<< 添加日志 >>>
    logger.debug(f"[{task_id_prefix}] 问题 Prompt 已生成，准备调用 API 获取问题...")
    # <<< 添加日志结束 >>>

    question = call_api(question_prompt, MODEL_NAME, TEMPERATURE_QUESTION, f"{task_id_prefix}-QGen", max_tokens=50)

    # <<< 添加日志 >>>
    logger.debug(f"[{task_id_prefix}] 问题 API 调用已返回 (问题内容: {'<生成失败或为空>' if not question else question[:50]+'...'})")
    # <<< 添加日志结束 >>>

    if not question or len(question) < 10:
        logger.error(f"[{task_id_prefix}] 问题生成失败或太短 (<10字)。跳过此组合。 Prompt: {question_prompt}")
        pbar.update(1)
        return False
    question = question.strip().strip('"').strip()
    logger.info(f"[{task_id_prefix}] 生成问题: {question}")

    num_related_to_generate = random.randint(NUM_RELATED_PRODUCTS_MIN, NUM_RELATED_PRODUCTS_MAX)
    logger.info(f"[{task_id_prefix}] 目标生成 {num_related_to_generate} 相关商品 和 {NUM_DISTRACTOR_PRODUCTS} 干扰商品。")

    related_product_ids = []
    distractor_product_ids = []
    success_flag = True # Assume success unless critical failure

    # --- Internal Function for Generating Products Concurrently ---
    def generate_products_internal(product_type, num_to_generate, prompt_builder_args, target_id_list):
        # (No changes needed inside generate_products_internal compared to v4.4)
        nonlocal success_flag
        type_prefix = "Rel" if product_type == "related" else "Dist"
        futures = {}
        generated_count = 0
        local_products_buffer = []
        last_successful_related_name = None

        if num_to_generate <= 0:
             logger.debug(f"[{task_id_prefix}-{type_prefix}] 商品无需生成 (数量 <= 0)。")
             return 0
        logger.debug(f"[{task_id_prefix}-{type_prefix}] 开始并发生成 {num_to_generate} 个商品...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix=f"{task_id_prefix}-{type_prefix}") as executor:
            for i in range(num_to_generate):
                task_id = f"{task_id_prefix}-{type_prefix}Gen-{i+1}"
                prompt = None
                current_prompt_args = prompt_builder_args.copy()

                if product_type == "related":
                    if last_successful_related_name:
                         current_prompt_args["previous_product_name"] = last_successful_related_name
                    prompt = generate_prompt("generate_related", **current_prompt_args)

                elif product_type == "distractor":
                    distractor_combos_list = current_prompt_args.get('distractor_combos')
                    if not distractor_combos_list:
                         logger.warning(f"[{task_id}] 干扰项组合列表为空，无法生成此干扰项。")
                         continue
                    distractor_combo = distractor_combos_list[i % len(distractor_combos_list)]
                    distractor_brand, distractor_line = distractor_combo
                    current_prompt_args.pop('distractor_combos', None)
                    current_prompt_args.update({
                         "category": category,
                         "brand": distractor_brand,
                         "product_line": distractor_line,
                         "original_question": question
                    })
                    prompt = generate_prompt("generate_specific_product", **current_prompt_args)
                else:
                    logger.error(f"[{task_id}] 未知的产品类型 '{product_type}'")
                    continue

                if prompt:
                    logger.debug(f"[{task_id}] 提交任务 {i+1}/{num_to_generate} (类型: {product_type})")
                    future = executor.submit(call_api, prompt, MODEL_NAME, TEMPERATURE_PRODUCT, task_id, max_tokens=350)
                    futures[future] = {"task_id": task_id, "used_previous_name": current_prompt_args.get("previous_product_name")}
                else:
                    logger.warning(f"[{task_id}] 未能为任务 {i+1}/{num_to_generate} 生成有效的 prompt。")


            for future in concurrent.futures.as_completed(futures):
                task_info = futures[future]
                task_id = task_info["task_id"]
                used_previous_name = task_info.get("used_previous_name")
                logger.debug(f"[{task_id}] Future 完成，开始处理结果...")

                try:
                    product_line_csv = future.result()
                    if product_line_csv:
                        logger.debug(f"[{task_id}] Future 返回非空结果，尝试解析...")
                        parsed_data = parse_product_csv_line(product_line_csv, task_id)
                        if parsed_data:
                            parsed_data[0] = parsed_data[0].strip().rstrip('"').strip()
                            current_product_name = parsed_data[0]

                            if product_type == "related":
                                if used_previous_name and current_product_name == used_previous_name:
                                     logger.warning(f"[{task_id}] 生成的相关商品名称 '{current_product_name}' 与上一个相同，尽管已提示避免。")
                                last_successful_related_name = current_product_name # Update for next related item in this batch

                            temp_id = get_unique_product_id()
                            product_entry = {"temp_id": temp_id, "data": parsed_data, "type": product_type}
                            local_products_buffer.append(product_entry)
                            target_id_list.append(temp_id)
                            generated_count += 1
                            logger.debug(f"[{task_id}] 商品处理成功 (Temp_ID: {temp_id}) Name: {current_product_name}")
                        else:
                             logger.warning(f"[{task_id}] 商品解析失败。 Raw: {product_line_csv[:100]}...")
                    else:
                         logger.warning(f"[{task_id}] Future 返回空结果 (API 调用可能失败或返回空)。")

                except Exception as exc:
                     logger.error(f"[{task_id}] 处理 future 结果时出错: {exc}", exc_info=True)

        logger.debug(f"[{task_id_prefix}-{type_prefix}] 并发生成结束，共获得 {generated_count} 个商品。")
        # Write buffer to global list and CSV file
        if local_products_buffer:
            logger.debug(f"[{task_id_prefix}-{type_prefix}] 将 {len(local_products_buffer)} 个商品写入全局列表和 CSV...")
            with product_list_lock:
                all_products_list.extend(local_products_buffer)
            for entry in local_products_buffer:
                 write_product_to_csv([entry["temp_id"]] + entry["data"], csv_writer)
            logger.debug(f"[{task_id_prefix}-{type_prefix}] 写入完成。")


        if product_type == "related" and generated_count == 0 and num_to_generate > 0 :
            logger.error(f"[{task_id_prefix}] 关键失败：未能成功生成任何 *相关* 商品！此组合处理失败。")
            success_flag = False
        elif generated_count < num_to_generate:
             logger.warning(f"[{task_id_prefix}] 成功生成 {generated_count}/{num_to_generate} 个 {product_type} 商品。")

        return generated_count
    # --- End of Internal Function ---

    # Generate Related Products
    logger.debug(f"[{task_id_prefix}] 开始生成相关商品...")
    related_args = {"question": question, "category": category, "brand": selected_brand, "product_line": selected_line}
    num_related_generated = generate_products_internal("related", num_related_to_generate, related_args, related_product_ids)
    logger.debug(f"[{task_id_prefix}] 相关商品生成过程结束 (成功生成 {num_related_generated} 个)。 Success Flag: {success_flag}")


    if success_flag:
        logger.debug(f"[{task_id_prefix}] 相关商品生成成功，开始生成干扰商品...")
        all_combos_in_category = set((b, l) for b in brands for l in product_lines if b and l)
        original_combo = (selected_brand, selected_line)
        distractor_combos = list(all_combos_in_category - {original_combo})

        num_distractors_generated = 0
        if not distractor_combos:
             logger.warning(f"[{task_id_prefix}] 无法为组合 {original_combo} 找到任何其他有效组合来生成干扰项！")
        elif NUM_DISTRACTOR_PRODUCTS > 0:
             logger.info(f"[{task_id_prefix}] 将从 {len(distractor_combos)} 个其他组合中选择生成 {NUM_DISTRACTOR_PRODUCTS} 个干扰项...")
             random.shuffle(distractor_combos)
             actual_distractors_to_generate = min(NUM_DISTRACTOR_PRODUCTS, len(distractor_combos) * 5)
             distractor_args = {"distractor_combos": distractor_combos}
             num_distractors_generated = generate_products_internal("distractor", actual_distractors_to_generate, distractor_args, distractor_product_ids)
             if num_distractors_generated == 0:
                  logger.warning(f"[{task_id_prefix}] 尝试生成干扰商品，但未能成功生成任何干扰商品。")
        else:
            logger.info(f"[{task_id_prefix}] 配置的干扰商品数量为 0，跳过生成。")
        logger.debug(f"[{task_id_prefix}] 干扰商品生成过程结束 (成功生成 {num_distractors_generated} 个)。")


    if success_flag:
        logger.debug(f"[{task_id_prefix}] 准备组装问题数据...")
        question_entry = {
            "question_id": task_id_prefix,
            "question": question,
            "category": category,
            "brand_context": selected_brand,
            "product_line_context": selected_line,
            "ground_truth_ids": related_product_ids,
            "distractor_ids": distractor_product_ids,
            "ground_truth_answer_points": [
                f"提及品牌 {selected_brand}",
                f"提及产品线 {selected_line}",
                "提供相关的具体商品信息 (型号/规格)",
                "给出商品价格",
                "说明商品库存状况"
            ]
        }
        with question_list_lock:
            test_questions_list.append(question_entry)
        logger.info(f"[{task_id_prefix}] 已保存问题及其关联 ID ({len(related_product_ids)} 相关, {len(distractor_product_ids)} 干扰)。")
    else:
         logger.error(f"[{task_id_prefix}] 由于关键步骤失败（未能生成相关商品），未保存此组合的问题集。")

    logger.debug(f"[{task_id_prefix}] 组合处理结束。")
    pbar.update(1)
    return success_flag

# --- Main Function ---
def main():
    """Main function to orchestrate data generation."""
    start_time_main = time.time()
    logger.info("="*20 + " 开始执行数据生成脚本 (v4.4d - Debug) " + "="*20)

    if not initialize_client():
        logger.critical("API 客户端初始化失败，脚本终止。请检查 .env 文件和网络连接。")
        return

    categories_available = list(CATEGORY_DETAILS.keys())
    logger.info(f"发现 {len(categories_available)} 个品类。")
    logger.info(f"配置: 每个品类最多尝试 {NUM_QUESTIONS_PER_CATEGORY} 个唯一组合的问题集。")
    logger.info(f"配置: 相关商品 {NUM_RELATED_PRODUCTS_MIN}-{NUM_RELATED_PRODUCTS_MAX} 个/问题。 (使用上个商品名提示多样性)")
    logger.info(f"配置: 干扰商品 {NUM_DISTRACTOR_PRODUCTS} 个/问题。")
    logger.info(f"配置: 商品数据写入: {OUTPUT_PRODUCT_CSV}")
    logger.info(f"配置: 问题数据写入: {OUTPUT_QUESTION_JSON}")
    logger.info(f"配置: 最大并发工作线程: {MAX_WORKERS}")


    output_csv_path = os.path.join(os.path.dirname(__file__), OUTPUT_PRODUCT_CSV)
    csvfile = None
    csv_writer = None
    try:
        logger.debug(f"尝试打开 CSV 文件进行写入: {output_csv_path}")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        csvfile = open(output_csv_path, 'w', newline='', encoding='utf-8-sig')
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # <<< 确认使用 Temp_ID >>>
        csv_writer.writerow(['Temp_ID', 'Name', 'Description', 'Price', 'Stock', 'Seller Username'])
        # <<< 结束 >>>
        logger.info(f"已创建/清空并写入 CSV 表头到 {output_csv_path}")
    except IOError as e:
        logger.error(f"无法打开/写入 CSV 文件 {output_csv_path}: {e}", exc_info=True)
        if csvfile:
             try: csvfile.close()
             except: pass
        return

    # --- Prepare Tasks based on Unique Combinations ---
    tasks = []
    total_tasks_planned = 0
    task_counter = 0

    logger.info("正在准备任务列表，计算并筛选唯一 (品牌, 产品线) 组合...")
    for category in categories_available:
        details = CATEGORY_DETAILS.get(category, {})
        brands = details.get("brands", [])
        product_lines = details.get("product_lines", [])

        if not brands: brands = ["通用品牌"]
        if not product_lines: product_lines = ["通用产品"]

        unique_combinations = list(set((b.strip(), l.strip()) for b in brands for l in product_lines if b.strip() and l.strip()))

        if not unique_combinations:
            logger.warning(f"品类 '{category}' 没有有效的 (品牌, 产品线) 组合，跳过此品类。")
            continue

        random.shuffle(unique_combinations)
        num_to_generate_for_category = min(NUM_QUESTIONS_PER_CATEGORY, len(unique_combinations))
        total_tasks_planned += num_to_generate_for_category

        if NUM_QUESTIONS_PER_CATEGORY > len(unique_combinations):
            logger.warning(f"品类 '{category}' 请求生成 {NUM_QUESTIONS_PER_CATEGORY} 个问题集，但只有 {len(unique_combinations)} 个唯一组合可用。将只生成 {len(unique_combinations)} 个。")
        elif num_to_generate_for_category > 0:
             logger.info(f"品类 '{category}' 有 {len(unique_combinations)} 个唯一组合，将为其生成 {num_to_generate_for_category} 个问题集。")

        for i in range(num_to_generate_for_category):
            selected_brand, selected_line = unique_combinations[i]
            task_counter += 1
            safe_brand = "".join(c if c.isalnum() else '_' for c in selected_brand)
            safe_line = "".join(c if c.isalnum() else '_' for c in selected_line)
            safe_category = "".join(c if c.isalnum() else '_' for c in category)
            task_id_suffix = f"{safe_category}-{safe_brand}-{safe_line}-{i+1}-{random.randint(100,999)}"[:60]
            tasks.append((category, details, csv_writer, selected_brand, selected_line, task_id_suffix))
            logger.debug(f"添加任务 {task_counter}: {task_id_suffix}")

    logger.info(f"总共计划生成 {total_tasks_planned} 个问题集 (基于唯一组合)。")
    est_related = (NUM_RELATED_PRODUCTS_MIN + NUM_RELATED_PRODUCTS_MAX) / 2.0
    total_products_estimate = total_tasks_planned * (est_related + NUM_DISTRACTOR_PRODUCTS)
    logger.info(f"预计生成总商品数: ~{total_products_estimate:.0f}")


    # --- Execute Tasks ---
    successful_tasks = 0
    failed_tasks = 0
    start_time_processing = time.time()
    logger.info(f"开始处理 {total_tasks_planned} 个任务...")

    with tqdm(total=total_tasks_planned, desc="Processing Combinations", unit="combo") as pbar:
        for task_params in tasks:
            category, details, writer, brand, line, task_id_suffix = task_params
            logger.debug(f"主循环：开始处理任务 {task_id_suffix}...")
            try:
                success = process_category_question_set(category, details, writer, pbar, brand, line, task_id_suffix)
                if success:
                    successful_tasks += 1
                    logger.debug(f"主循环：任务 {task_id_suffix} 成功完成。")
                else:
                    failed_tasks += 1
                    logger.debug(f"主循环：任务 {task_id_suffix} 失败。")
            except Exception as exc:
                failed_tasks += 1
                logger.error(f"任务 {task_id_suffix} ({category}/{brand}/{line}) 产生顶层异常: {exc}", exc_info=True)
                pbar.update(1) # Ensure progress bar updates on error


    # --- Finalization ---
    processing_duration = time.time() - start_time_processing
    logger.info(f"任务处理完成，耗时: {processing_duration:.2f} 秒。")

    if csvfile:
        try:
            logger.debug(f"尝试关闭 CSV 文件: {output_csv_path}")
            csvfile.close()
            logger.info(f"CSV 文件 {output_csv_path} 已关闭。")
        except Exception as e:
            logger.error(f"关闭 CSV 文件时出错: {e}", exc_info=True)

    logger.info("\n" + "="*30 + " 数据生成总结 " + "="*30)
    logger.info(f"成功完成 {successful_tasks} / {total_tasks_planned} 个唯一组合的处理。")
    if failed_tasks > 0:
        logger.warning(f"有 {failed_tasks} 个组合未能成功完成所有数据生成步骤 (详见以上日志)。")

    output_json_path = os.path.join(os.path.dirname(__file__), OUTPUT_QUESTION_JSON)
    try:
        logger.debug(f"尝试写入 JSON 文件: {output_json_path}")
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        test_questions_list.sort(key=lambda x: x.get('question_id', ''))
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(test_questions_list, f, ensure_ascii=False, indent=2)
        logger.info(f"问题及 ID 关联已保存到: {output_json_path}")
        # <<< 确认使用 Temp_ID >>>
        logger.info(f"全局商品 Temp_ID 计数器达到: {product_id_counter} (这代表尝试生成的最大 Temp_ID)。")
        # <<< 结束 >>>
        logger.info(f"总共生成了 {len(test_questions_list)} 个测试问题条目。")
        logger.info(f"请检查 {output_csv_path} 和 {output_json_path} 文件确认最终结果。")
    except IOError as e:
        logger.error(f"保存问题 JSON 文件时出错 ({output_json_path}): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"保存问题 JSON 时发生未知错误: {e}", exc_info=True)

    total_duration = time.time() - start_time_main
    logger.info(f"脚本总执行时间: {total_duration:.2f} 秒")
    logger.info("="*70)

# --- Script Entry Point ---
if __name__ == "__main__":
    logger.debug("脚本入口点 main() 即将执行。")
    main()
    logger.debug("脚本 main() 执行完毕。")