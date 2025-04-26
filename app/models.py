# app/models.py (Added ProductImage model and relationship)
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
from typing import Optional, List # 使用 Optional/List 兼容 Python 3.9
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

    # --- Product relationship (在 Product 定义后通过 backref 添加) ---
    # products = db.relationship(...) <- 通过 Product.seller 添加

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if self.password_hash is None: return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

# --- 商品模型 (增加 images 关系) ---
class Product(db.Model):
    """商品数据模型"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, default=0)
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='active', index=True)
    # image_url = db.Column(db.String(255), nullable=True) # Day 11 后应移除或弃用
    embedding = db.Column(db.LargeBinary, nullable=True) # 存储序列化向量

    seller = db.relationship('User', backref=db.backref('products', lazy=True))
    # --- vvv 添加与 ProductImage 的关系 vvv ---
    # cascade='all, delete-orphan' 意味着删除 Product 时，所有关联的 ProductImage 也会被自动删除
    # 这可以简化 admin.py 中的删除逻辑，但我们已在那里手动处理，保留手动处理更明确
    images = db.relationship('ProductImage', backref='product', lazy='dynamic', cascade='all, delete-orphan')
    # --- ^^^ 添加与 ProductImage 的关系 ^^^ ---


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

    # --- vvv 新增：获取主图片（或第一张图片）的方法 vvv ---
    @property
    def main_image_filename(self) -> Optional[str]:
        """获取此商品的第一张关联图片的文件名 (如果存在)"""
        first_image = self.images.first() # lazy='dynamic' 允许使用 .first()
        return first_image.image_filename if first_image else None
    # --- ^^^ 新增 ^^^ ---

    def __repr__(self):
        return f'<Product {self.name}>'

# --- vvv 新增：商品图片模型 vvv ---
class ProductImage(db.Model):
    """存储商品图片信息"""
    __tablename__ = 'product_images' # 建议明确指定表名
    id = db.Column(db.Integer, primary_key=True)
    # 外键，关联到 product 表的 id 字段
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False, index=True)
    # 存储图片的文件名 (相对于 UPLOAD_FOLDER)
    image_filename = db.Column(db.String(255), nullable=False)
    # (可选) 可以添加 'is_main' 字段来标记主图，或 'order' 字段来排序

    # product 关系通过 Product.images 的 backref='product' 自动建立

    def __repr__(self):
        return f'<ProductImage {self.image_filename} for Product ID {self.product_id}>'
# --- ^^^ 新增 ^^^ ---


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

# --- 系统设置模型 (保持不变) ---
class SystemSetting(db.Model):
    """存储系统级键值对设置"""
    __tablename__ = 'system_setting' # 明确表名
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False, index=True)
    value = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<SystemSetting {self.key}={self.value}>'


# --- Flask-Login 回调函数 (保持不变) ---
@login_manager.user_loader
def load_user(user_id):
    """根据存储的用户ID加载用户对象"""
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None