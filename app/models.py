# app/models.py
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
# 从同级目录的 __init__.py 文件导入 db 和 login_manager 实例
from . import db, login_manager

# --- 用户模型 (保持不变) ---
class User(UserMixin, db.Model):
    """用户数据模型 (蓝图)"""
    id = db.Column(db.Integer, primary_key=True) # 用户编号，独一无二 (主键)
    username = db.Column(db.String(80), unique=True, nullable=False) # 用户名，不能重复，不能为空
    email = db.Column(db.String(120), unique=True, nullable=False) # 邮箱，不能重复，不能为空
    password_hash = db.Column(db.String(256)) # 存储加密后的密码，设长一点安全
    role = db.Column(db.String(10), nullable=False, default='buyer') # 角色 ('buyer', 'seller', 'admin')，默认是买家

    # 密码处理方法 (保持不变)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

# --- 商品模型 (保持不变) ---
class Product(db.Model):
    """商品数据模型 (蓝图)"""
    id = db.Column(db.Integer, primary_key=True) # 商品编号
    name = db.Column(db.String(100), nullable=False) # 商品名称，不能为空
    description = db.Column(db.Text, nullable=True) # 商品描述，可以为空
    price = db.Column(db.Float, nullable=False) # 价格，不能为空
    stock = db.Column(db.Integer, default=0) # 库存，默认为0
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='active', index=True)
    image_url = db.Column(db.String(255), nullable=True)
    seller = db.relationship('User', backref=db.backref('products', lazy=True))

    @property
    def is_active(self):
        return self.status == 'active'

    def __repr__(self):
        return f'<Product {self.name}>'

# --- 规则/FAQ 模型 (保持不变) ---
class FaqRule(db.Model):
    """规则/FAQ 数据模型"""
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), index=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<FaqRule {self.category}: {self.question[:30]}...>'


# === vvv 新增：AI 模型配置模型 vvv ===
class AiModel(db.Model):
    """存储可用 AI 模型及其状态的模型"""
    id = db.Column(db.Integer, primary_key=True)
    # 模型名称 (例如 'gpt-4.1', 'gpt-3.5-turbo')，这是传递给 OpenAI API 的实际名称
    model_name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    # 显示名称 (可选，用于在界面上更友好地展示)
    display_name = db.Column(db.String(100), nullable=True)
    # 是否为当前活动模型 (理论上应该只有一个是 True)
    is_active = db.Column(db.Boolean, default=False, nullable=False, index=True)
    # (可选) 添加描述字段
    description = db.Column(db.Text, nullable=True)
    # (可选) 记录添加时间
    # created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        active_status = "[Active]" if self.is_active else ""
        return f'<AiModel {self.model_name} ({self.display_name or "N/A"}) {active_status}>'
# === ^^^ 新增：AI 模型配置模型 ^^^ ===


# --- Flask-Login 回调函数 (保持不变) ---
@login_manager.user_loader
def load_user(user_id):
    """根据存储的用户ID加载用户对象，供 Flask-Login 内部使用"""
    try:
        user_id_int = int(user_id)
        return User.query.get(user_id_int)
    except (ValueError, TypeError):
        return None