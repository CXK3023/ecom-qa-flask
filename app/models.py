# app/models.py (Refactored + Re-added SystemSetting)
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
from typing import Optional # 使用 Optional 兼容 Python 3.9
# 从同级目录的 __init__.py 文件导入 db 和 login_manager 实例
from . import db, login_manager

# --- 用户模型 (保持不变) ---
class User(UserMixin, db.Model):
    """用户数据模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(10), nullable=False, default='buyer') # 'buyer', 'seller', 'admin'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if self.password_hash is None: return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

# --- 商品模型 (保持不变) ---
class Product(db.Model):
    """商品数据模型"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, default=0)
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='active', index=True)
    image_url = db.Column(db.String(255), nullable=True)
    embedding = db.Column(db.LargeBinary, nullable=True) # 存储序列化向量

    seller = db.relationship('User', backref=db.backref('products', lazy=True))

    @property
    def is_active(self):
        return self.status == 'active'

    @property
    def vector(self) -> Optional[np.ndarray]:
        """将存储的二进制数据反序列化为 NumPy 向量"""
        if self.embedding:
            try:
                vec = pickle.loads(self.embedding)
                return vec if isinstance(vec, np.ndarray) else None
            except Exception:
                return None
        return None

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

# --- 对话模型 (保持不变) ---
class AiModel(db.Model):
    """存储可用 **对话 (Chat)** 模型及其状态的模型"""
    __tablename__ = 'ai_model'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(100), nullable=True)
    is_active = db.Column(db.Boolean, default=False, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        active_status = "[Active Chat]" if self.is_active else ""
        return f'<AiModel {self.model_name} ({self.display_name or "N/A"}) {active_status}>'

# --- 向量嵌入模型 (保持不变) ---
class EmbeddingModel(db.Model):
    """存储可用 **向量嵌入 (Embedding)** 模型及其状态的模型"""
    __tablename__ = 'embedding_model'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(100), nullable=True)
    is_active = db.Column(db.Boolean, default=False, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    invocation_method = db.Column(db.String(20), nullable=False, default='local', index=True)

    def __repr__(self):
        active_status = "[Active Embedding]" if self.is_active else ""
        method = f"({self.invocation_method})"
        return f'<EmbeddingModel {self.model_name} ({self.display_name or "N/A"}) {method} {active_status}>'

# === vvv 重新添加：系统设置模型 vvv ===
class SystemSetting(db.Model):
    """存储系统级键值对设置"""
    __tablename__ = 'system_setting' # 明确表名
    id = db.Column(db.Integer, primary_key=True)
    # 设置项的键名，例如 'enable_vector_search'
    key = db.Column(db.String(50), unique=True, nullable=False, index=True)
    # 设置项的值，存储为字符串 (例如 'true'/'false')
    value = db.Column(db.String(255), nullable=True)
    # (可选) 添加描述字段
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<SystemSetting {self.key}={self.value}>'
# === ^^^ 重新添加 ^^^ ===


# --- Flask-Login 回调函数 (保持不变) ---
@login_manager.user_loader
def load_user(user_id):
    """根据存储的用户ID加载用户对象"""
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None