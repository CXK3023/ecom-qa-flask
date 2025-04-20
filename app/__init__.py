# This file is intentionally left blank.
# app/__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件里的“秘密”

# 创建数据库和登录管理的“管家”实例
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login' # 如果没登录，告诉他去哪个页面登录 (后面会创建auth蓝图)

def create_app():
    """创建并配置 Flask 应用实例"""
    app = Flask(__name__)

    # 从 .env 文件加载配置
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # 关闭一个通常不需要的功能

    # 初始化“管家”
    db.init_app(app)
    login_manager.init_app(app)

    # 注册蓝图 (后面会添加具体的页面路由)
    from .routes.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    from .routes.main import main as main_blueprint # [source: 41]
    app.register_blueprint(main_blueprint) # main 蓝图不需要 URL 前缀 # [source: 41] 

    # 确保在应用上下文中创建数据库表（如果尚不存在）
    # 这不是最佳实践，后面会用 flask shell 或 Flask-Migrate 替代
    # with app.app_context():
    #     db.create_all()

    return app