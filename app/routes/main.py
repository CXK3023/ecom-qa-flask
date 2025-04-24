# app/routes/main.py
from flask import Blueprint, render_template, abort, request, flash, redirect, url_for # 确保导入齐全
from flask_login import login_required, current_user
from ..models import Product, User # 确保导入 User
from .. import db # 确保导入 db
import re

main = Blueprint('main', __name__)

# 简单的邮箱格式正则表达式
EMAIL_REGEX = re.compile(r'[^@]+@[^@]+\.[^@]+')

# --- 首页路由 ---
@main.route('/')
def index():
    """网站首页，显示所有商品列表"""
    all_products = Product.query.filter_by(status='active').order_by(Product.id.desc()).all()
    return render_template('main/index.html', products=all_products)

# --- 用户个人资料页路由 ---
@main.route('/profile')
@login_required
def profile():
    """用户个人资料页"""
    return render_template('main/profile.html', user=current_user)

# --- 商品详情页路由 ---
@main.route('/product/<int:product_id>')
def product_detail(product_id):
    """显示单个商品的详情"""
    product = Product.query.get_or_404(product_id)
    return render_template('main/product_detail.html', product=product)


# --- 修改邮箱路由 ---
@main.route('/change-email', methods=['GET', 'POST'])
@login_required
def change_email():
    """处理用户修改邮箱地址"""
    if request.method == 'POST':
        new_email = request.form.get('new_email')
        current_password = request.form.get('current_password')
        error = None

        if not current_user.check_password(current_password):
            error = '当前密码不正确，无法修改邮箱。'
        elif not new_email:
            error = '新邮箱地址不能为空。'
        elif not EMAIL_REGEX.match(new_email):
             error = '请输入有效的邮箱地址格式。'
        elif new_email == current_user.email:
            error = '新邮箱地址不能与当前邮箱地址相同。'
        else:
            existing_user = User.query.filter(User.email == new_email, User.id != current_user.id).first()
            if existing_user:
                error = f'邮箱 "{new_email}" 已被其他用户注册，请选择其他邮箱。'

        if error is None:
            try:
                current_user.email = new_email
                db.session.commit()
                flash('邮箱地址已成功更新！', 'success')
                return redirect(url_for('main.profile'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新邮箱时发生错误: {e}', 'error')
                print(f"ERROR updating email for user {current_user.id}: {e}")
        else:
            flash(error, 'error')

    return render_template('main/change_email.html')


# === vvv 修改后的 change_username 函数 vvv ===
@main.route('/change-username', methods=['GET', 'POST'])
@login_required
def change_username():
    """处理用户修改用户名"""
    if request.method == 'POST':
        new_username = request.form.get('new_username')
        current_password = request.form.get('current_password')
        error = None

        # 1. 验证当前密码
        if not current_user.check_password(current_password):
            error = '当前密码不正确，无法修改用户名。'

        # 2. 验证新用户名
        elif not new_username:
            error = '新用户名不能为空。'
        elif new_username == current_user.username:
            error = '新用户名不能与当前用户名相同。'
        # (可选) 在这里可以添加更多用户名规则校验，例如长度、不允许特殊字符等
        else:
            # 3. 检查新用户名是否已被其他用户占用
            existing_user = User.query.filter(User.username == new_username, User.id != current_user.id).first()
            if existing_user:
                error = f'用户名 "{new_username}" 已被占用，请选择其他用户名。'

        # 4. 如果所有验证通过
        if error is None:
            try:
                current_user.username = new_username # 更新当前用户的用户名属性
                db.session.commit() # 保存到数据库
                flash('用户名已成功更新！', 'success')
                # 用户名更新后，导航栏等地方会立刻显示新用户名
                return redirect(url_for('main.profile')) # 重定向回个人资料页
            except Exception as e:
                db.session.rollback() # 出错时回滚
                flash(f'更新用户名时发生错误: {e}', 'error')
                print(f"ERROR updating username for user {current_user.id}: {e}") # 记录错误日志
        else:
            # 如果有错误，显示错误信息
            flash(error, 'error')
            # 保持在当前页面

    # GET 请求或 POST 请求有错误时，显示修改用户名的表单
    return render_template('main/change_username.html')
# === ^^^ 修改后的 change_username 函数 ^^^ ===