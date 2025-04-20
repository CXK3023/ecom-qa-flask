# app/routes/main.py
from flask import Blueprint, render_template
from flask_login import login_required, current_user # [source: 39]

main = Blueprint('main', __name__) # [source: 39]

@main.route('/') # 网站根路径 # [source: 39]
def index():
    """网站首页""" # [source: 39]
    return render_template('main/index.html') # [source: 39]

# 举例：一个需要登录才能访问的页面
@main.route('/profile') # [source: 39]
@login_required # 添加这个装饰器，访问此页面需要登录 # [source: 39]
def profile():
    """用户个人资料页 (示例)""" # [source: 39]
    # 可以在模板里使用 current_user 获取用户信息
    return render_template('main/profile.html', user=current_user) # [source: 39]