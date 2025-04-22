# app/routes/main.py
from flask import Blueprint, render_template, abort # 确保导入了 abort
from flask_login import login_required, current_user # [source: 39]
from ..models import Product # 导入 Product 模型 [source: 36]

main = Blueprint('main', __name__) # [source: 39]

# --- 修改后的首页路由 ---
@main.route('/') # 网站根路径 # [source: 39]
def index():
    """网站首页，显示所有商品列表""" # [source: 36]
   # 查询数据库中所有状态为 'active' 的商品, 按 ID 降序排列
    all_products = Product.query.filter_by(status='active').order_by(Product.id.desc()).all()
    # 把查询到的所有商品列表传递给模板，模板里用 'products' 这个名字来接收
    return render_template('main/index.html', products=all_products) # [source: 36]

# --- 保留的用户个人资料页路由 ---
@main.route('/profile') # [source: 39]
@login_required # 添加这个装饰器，访问此页面需要登录 # [source: 39]
def profile():
    """用户个人资料页 (示例)""" # [source: 39]
    # 可以在模板里使用 current_user 获取用户信息
    return render_template('main/profile.html', user=current_user) # [source: 39]

# --- 第七步会添加的商品详情页路由 ---
@main.route('/product/<int:product_id>') # [source: 39]
def product_detail(product_id):
    """显示单个商品的详情""" # [source: 39]
    # 使用 get_or_404 获取商品，如果 ID 无效则自动返回 404 Not Found 页面
    product = Product.query.get_or_404(product_id) # [source: 39]
    # 将商品对象传递给模板
    return render_template('main/product_detail.html', product=product) # [source: 39]