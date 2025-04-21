# app/__init__.py
import os
import re
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv
# === 修改：从 markupsafe 导入 Markup 和 escape ===
from markupsafe import Markup, escape

load_dotenv()

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = "请先登录以访问此页面。"
login_manager.login_message_category = "info"

# 定义 nl2br 过滤器函数
_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')

# 保持这个函数定义不变
def nl2br(value):
    """将纯文本中的换行符转换成 HTML 的 <br> 标签。"""
    result = u'\n\n'.join(u'<p>%s</p>' % p.replace('\n', Markup('<br>\n'))
                          for p in _paragraph_re.split(escape(value)))
    return Markup(result)

def create_app():
    """创建并配置 Flask 应用实例"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key-for-dev')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 注册自定义过滤器 (这行保持不变)
    app.jinja_env.filters['nl2br'] = nl2br

    db.init_app(app)
    login_manager.init_app(app)

    # --- 注册蓝图 ---
    from .routes.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    from .routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    from .routes.seller import seller as seller_blueprint
    app.register_blueprint(seller_blueprint, url_prefix='/seller')
    from .routes.qa import qa as qa_blueprint
    app.register_blueprint(qa_blueprint, url_prefix='/qa')
    # --- 蓝图注册结束 ---

    return app