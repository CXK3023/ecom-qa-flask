# This file is intentionally left blank.
# app/models.py
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
# 从同级目录的 __init__.py 文件导入 db 和 login_manager 实例
from . import db, login_manager

# --- 用户模型 ---
class User(UserMixin, db.Model):
    """用户数据模型 (蓝图)"""
    id = db.Column(db.Integer, primary_key=True) # 用户编号，独一无二 (主键)
    username = db.Column(db.String(80), unique=True, nullable=False) # 用户名，不能重复，不能为空
    email = db.Column(db.String(120), unique=True, nullable=False) # 邮箱，不能重复，不能为空
    password_hash = db.Column(db.String(256)) # 存储加密后的密码，设长一点安全
    role = db.Column(db.String(10), nullable=False, default='buyer') # 角色 ('buyer' 或 'seller')，默认是买家

    def set_password(self, password):
        """设置密码的方法，会自动加密"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """检查输入的密码是否和存储的加密密码匹配"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        # 这个方法方便我们在调试时打印用户信息，看到的是用户名
        return f'<User {self.username}>'

# --- 商品模型 ---
class Product(db.Model):
    """商品数据模型 (蓝图)"""
    id = db.Column(db.Integer, primary_key=True) # 商品编号
    name = db.Column(db.String(100), nullable=False) # 商品名称，不能为空
    description = db.Column(db.Text, nullable=True) # 商品描述，可以为空
    price = db.Column(db.Float, nullable=False) # 价格，不能为空
    stock = db.Column(db.Integer, default=0) # 库存，默认为0
    # 外键: 关联到卖家的用户ID，表示这个商品是谁卖的
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # 'user.id' 指向 User 模型的 id 字段

    # 关系: 建立 User 和 Product 之间的联系
    # 这样可以通过 product.seller 找到卖家用户,
    # 或者通过 user.products 找到该用户的所有商品
    seller = db.relationship('User', backref=db.backref('products', lazy=True))

    def __repr__(self):
        return f'<Product {self.name}>'

# --- Flask-Login 需要的回调函数 ---
@login_manager.user_loader
def load_user(user_id):
    """根据存储的用户ID加载用户对象，供 Flask-Login 内部使用"""
    # User.query.get 是 SQLAlchemy 提供的方法，通过主键(id)查找用户
    return User.query.get(int(user_id))

# --- (可选) 规则/FAQ 模型 ---
# class FaqRule(db.Model):
#     """规则/FAQ 数据模型 (第5天会用到)"""
#     id = db.Column(db.Integer, primary_key=True)
#     category = db.Column(db.String(50)) # 分类 (如 '售后', '商家操作')
#     question = db.Column(db.Text, nullable=False) # 问题
#     answer = db.Column(db.Text, nullable=False) # 答案
#
#     def __repr__(self):
#         return f'<FaqRule {self.category}: {self.question[:30]}>'