# app/__init__.py
import os
import re
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv
from markupsafe import Markup, escape # 保持这个导入

load_dotenv()

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = "请先登录以访问此页面。"
login_manager.login_message_category = "info"

# --- nl2br 过滤器 (保持不变) ---
_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')

def nl2br(value):
    """将纯文本中的换行符转换成 HTML 的 <br> 标签。"""
    if value is None:
        return ''
    escaped_value = escape(value)
    paragraphs = _paragraph_re.split(escaped_value)
    result_html = u'\n\n'.join(u'<p>%s</p>' % p.replace('\n', Markup('<br>\n'))
                              for p in paragraphs if p)
    return Markup(result_html)
# --- nl2br 过滤器结束 ---

def create_app():
    """创建并配置 Flask 应用实例"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key-for-dev')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # === vvv NEW: 文件上传配置 vvv ===
    # 获取当前文件 (__init__.py) 所在的目录 (即 app 目录)
    basedir = os.path.abspath(os.path.dirname(__file__))
    # 定义上传文件夹的路径 (app/static/uploads/products)
    # 我们将图片存储在 static 目录下，这样 Flask 可以直接提供访问
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads', 'products')
    # 定义允许上传的文件扩展名
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # 将配置添加到 app.config
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
    # (可选) 限制上传文件的大小，例如 16MB
    # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    print(f"DEBUG: UPLOAD_FOLDER set to: {app.config['UPLOAD_FOLDER']}") # 添加日志确认路径
    # === ^^^ NEW: 文件上传配置结束 ^^^ ===


    # 注册自定义过滤器 (保持不变)
    app.jinja_env.filters['nl2br'] = nl2br

    db.init_app(app)
    login_manager.init_app(app)

    # --- 注册蓝图 (保持不变) ---
    from .routes.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    from .routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    from .routes.seller import seller as seller_blueprint
    app.register_blueprint(seller_blueprint, url_prefix='/seller')
    from .routes.qa import qa as qa_blueprint
    app.register_blueprint(qa_blueprint, url_prefix='/qa')
    from .routes.admin import admin as admin_blueprint
    app.register_blueprint(admin_blueprint, url_prefix='/admin')

    # --- 蓝图注册结束 ---

    return app