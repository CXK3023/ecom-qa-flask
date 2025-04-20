# app/routes/auth.py
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from ..models import User # 从上一级目录的 models.py 导入 User 模型
from .. import db # 从 app/__init__.py 导入 db

# 创建一个名叫 'auth' 的蓝图
auth = Blueprint('auth', __name__)

# --- 后面我们会在这里添加注册、登录、退出的路由 ---

# app/routes/auth.py (替换原来的 register 函数)
@auth.route('/register', methods=['GET', 'POST']) # 允许 GET 和 POST 请求 # [source: 27]
def register():
    """处理用户注册""" # [source: 28]
    if current_user.is_authenticated: # [source: 28]
        # 如果用户已经登录，直接跳转到首页
        return redirect(url_for('main.index')) # [source: 28]

    if request.method == 'POST': # [source: 28]
        # 如果是 POST 请求（用户提交了表单）
        username = request.form.get('username') # [source: 28]
        email = request.form.get('email') # [source: 28]
        password = request.form.get('password') # [source: 28]
        password2 = request.form.get('password2') # [source: 28]
        role = request.form.get('role') # [source: 29]

        # --- 基本验证 ---
        user_by_username = User.query.filter_by(username=username).first() # 检查用户名是否已存在 # [source: 29]
        user_by_email = User.query.filter_by(email=email).first() # 检查邮箱是否已存在 # [source: 29]
        error = None # [source: 29]

        if not username: # [source: 29]
            error = '用户名不能为空。' # [source: 29]
        elif not email: # [source: 29]
            error = '邮箱不能为空。' # [source: 30]
        elif not password: # [source: 30]
            error = '密码不能为空。' # [source: 30]
        elif password != password2: # [source: 30]
            error = '两次输入的密码不一致。' # [source: 30]
        elif user_by_username: # [source: 30]
            error = f'用户名 "{username}" 已被注册。' # [source: 30]
        elif user_by_email: # [source: 30]
            error = f'邮箱 "{email}" 已被注册。' # [source: 31]
        elif role not in ['buyer', 'seller']: # [source: 31]
            error = '无效的角色选择。' # [source: 31]

        if error is None: # [source: 31]
            # 如果没有错误，创建新用户
            new_user = User(username=username, email=email, role=role) # [source: 31]
            new_user.set_password(password) # 加密密码 # [source: 31]
            try: # [source: 32]
                db.session.add(new_user) # 将新用户添加到数据库会话 # [source: 32]
                db.session.commit() # 提交更改到数据库 # [source: 32]
            except Exception as e: # [source: 32]
                db.session.rollback() # 如果出错，回滚更改 # [source: 32]
                flash(f'注册过程中发生错误: {e}', 'error') # [source: 33]
            else: # [source: 33]
                flash('恭喜！你已成功注册，请登录。', 'success') # [source: 33]
                return redirect(url_for('auth.login')) # 注册成功后跳转到登录页面 # [source: 33]
        else:
            # 如果有错误，显示错误信息
            flash(error, 'error') # [source: 33]

    # 如果是 GET 请求（首次访问页面）或者 POST 请求处理中有错误，就显示注册表单
    return render_template('auth/register.html') # [source: 34]


# app/routes/auth.py (替换原来的 login 函数)
@auth.route('/login', methods=['GET', 'POST']) # [source: 36]
def login():
    """处理用户登录""" # [source: 36]
    if current_user.is_authenticated: # [source: 36]
        return redirect(url_for('main.index')) # 已登录用户直接去首页 # [source: 36]

    if request.method == 'POST': # [source: 36]
        email = request.form.get('email') # [source: 36]
        password = request.form.get('password') # [source: 36]
        error = None # [source: 36]
        user = User.query.filter_by(email=email).first() # 根据邮箱查找用户 # [source: 37]

        if user is None: # [source: 37]
            error = '邮箱地址不存在。' # [source: 37]
        elif not user.check_password(password): # 使用我们模型里的 check_password 方法验证密码 # [source: 37]
            error = '密码不正确。' # [source: 37]

        if error is None: # [source: 37]
            # 密码正确，执行登录
            login_user(user) # Flask-Login 提供的登录函数，会记录用户登录状态 # [source: 38]
            flash('登录成功！', 'success') # [source: 38]
            # 尝试跳转到用户之前想访问的页面，如果没有就去首页
            next_page = request.args.get('next') # [source: 38]
            return redirect(next_page or url_for('main.index')) # [source: 38]
        else:
            flash(error, 'error') # [source: 38]

    return render_template('auth/login.html') # [source: 38]

@auth.route('/logout')
@login_required # 必须登录才能访问这个页面
def logout():
    # 暂时先返回一个简单的文本
    logout_user() # Flask-Login 提供的退出登录函数
    flash('你已成功退出登录。') # 提示信息
    return redirect(url_for('main.index')) # 重定向到首页 (main.index 稍后创建) # [source: 11]