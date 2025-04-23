# 电商智能问答系统 (Flask + MySQL + OpenAI)

## 1. 项目简介

本项目旨在利用 Flask 框架、MySQL 数据库以及通过代理网关调用的 OpenAI 大语言模型，构建一个面向电商领域的智能问答系统原型 [cite: 98]。

**核心功能包括:**

* **用户认证**: 支持买家和商家注册、登录、退出 [cite: 9]。
* **商品管理 (商家)**: 商家可以登录后添加、编辑、删除自己发布的商品 [cite: 9, 16]。
* **商品浏览 (用户)**: 所有用户可以浏览首页展示的商品列表和查看商品详情 [cite: 22, 23]。
* **智能问答**:
    * 集成 OpenAI API，提供基础问答能力 [cite: 26]。
    * 具备基于规则的问答能力，AI 会参考预设的平台规则和 FAQ 进行回答 [cite: 57, 65]。
    * 具备场景感知能力，能根据用户所在的页面（如商品详情页、商家后台）提供更相关的回答 [cite: 57, 73, 78]。
* **后台管理 (管理员)**: 管理员拥有专属后台，可以管理平台规则 (FAQ) [cite: 92, 94]、管理所有用户 (查看、修改角色、重置密码、删除) [cite: 96] 以及管理所有商品 (查看、编辑、切换状态、删除) [cite: 96, 97]。

## 2. 技术栈

* **后端框架**: Flask [cite: 98]
* **数据库**: MySQL 8.0 (通过 Docker 运行) [cite: 98, 100]
* **ORM**: Flask-SQLAlchemy [cite: 98]
* **数据库连接器**: mysql-connector-python [cite: 4, 101] (或 PyMySQL)
* **用户认证**: Flask-Login [cite: 4, 98]
* **LLM API**: OpenAI Python library (`openai` v1.x+) [cite: 26, 98]
* **API 代理**: 通过自定义网关调用 OpenAI API (在 `.env` 中配置 `OPENAI_API_BASE`) [cite: 26, 98, 114]
* **Python 环境**: Conda (推荐 Python 3.9+) [cite: 2, 98, 99]
* **前端**: HTML, Bootstrap 5, JavaScript [cite: 92, 98]
* **配置管理**: python-dotenv (`.env` 文件) [cite: 4, 98]
* **容器化**: Docker, Docker Compose [cite: 3, 98]
* **开发环境推荐**: VS Code + Remote - WSL 扩展 (连接到 WSL2 Ubuntu) [cite: 1, 98]

## 3. 项目结构

```

ecom-qa-flask/
├── app/                     \# Flask 应用核心目录
│   ├── **init**.py          \# 应用工厂, 初始化扩展, 注册蓝图, 定义过滤器
│   ├── models.py            \# SQLAlchemy 数据模型 (User, Product, FaqRule) [cite: 102]
│   ├── routes/              \# 存放路由蓝图 [cite: 102]
│   │   ├── **init**.py
│   │   ├── auth.py          \# 认证路由 (/auth/register, /auth/login, ...) [cite: 103]
│   │   ├── main.py          \# 主要页面路由 (/, /profile, /product/\<id\>) [cite: 103]
│   │   ├── seller.py        \# 商家后台路由 (/seller/dashboard, ...) [cite: 103]
│   │   ├── admin.py         \# 管理员后台路由 (/admin/rules, /admin/users, ...) [cite: 92, 96]
│   │   └── qa.py            \# 问答 API 路由 (/qa/ask) [cite: 103]
│   ├── services/            \# 业务逻辑和服务 [cite: 104]
│   │   ├── **init**.py
│   │   ├── openai\_service.py \# 封装 OpenAI API 调用 [cite: 104]
│   │   └── data\_service.py   \# 封装规则/FAQ 数据访问 (数据库交互) [cite: 92, 104]
│   ├── static/              \# 静态文件 (CSS, JS, Images) - 本项目主要使用 CDN [cite: 104]
│   └── templates/           \# Jinja2 HTML 模板 [cite: 105]
│       ├── layouts/         \# 基础布局模板 (base.html) [cite: 105]
│       ├── auth/            \# 认证相关模板 (login.html, register.html) [cite: 105]
│       ├── main/            \# 主要页面模板 (index.html, profile.html, product\_detail.html) [cite: 105, 106]
│       ├── seller/          \# 商家后台模板 (dashboard.html, add\_product.html, ...) [cite: 106]
│       ├── admin/           \# 管理员后台模板 (rules\_list.html, users\_list.html, ...) [cite: 92, 96]
│       └── partials/        \# (可选) 可重用的模板片段
├── data/                    \# (历史遗留) 存放过 rules.json [cite: 107]
├── migrations/              \# (未使用) Flask-Migrate 目录
├── tests/                   \# (未使用) 测试目录
├── .env                     \# 环境变量 (\!\!\! 重要: 不应提交到 Git \!\!\!) [cite: 108]
├── .gitignore               \# Git 忽略配置 [cite: 109]
├── docker-compose.yml       \# Docker 配置 [cite: 109]
├── requirements.txt         \# Python 依赖列表 (由 pip freeze 生成) [cite: 109]
└── run.py                   \# Flask 应用启动脚本 [cite: 109]

````

## 4. 环境搭建 (WSL2 Ubuntu 推荐)

1.  **安装核心工具**:
    * VS Code [cite: 1]
    * WSL2 (例如 Ubuntu 20.04 或 22.04) [cite: 1]
    * Docker Desktop (确保在设置中启用了 WSL2 集成) [cite: 1, 98]
    * Git [cite: 1]
    * Miniconda 或 Anaconda [cite: 1]

2.  **VS Code 连接 WSL**:
    * 安装 VS Code 的 "Remote - WSL" 扩展 [cite: 98]。
    * 点击 VS Code 左下角的绿色图标，选择连接到你的 WSL 发行版 [cite: 1]。
    * 在 VS Code 中打开 WSL 终端 (Terminal -> New Terminal)。后续所有命令都在此终端执行。

3.  **获取代码**:
    * `git clone <your-repository-url>` (如果代码在 Git 仓库中)
    * 或者直接将 `ecom-qa-flask` 文件夹复制到 WSL 环境中。
    * `cd ecom-qa-flask` 进入项目根目录。

4.  **创建并激活 Conda 环境**:
    ```bash
    conda create -n ecom_qa python=3.9 -y
    conda activate ecom_qa
    ```
    [cite: 2, 99]

5.  **安装 Python 依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    (确保 `requirements.txt` 文件存在于项目根目录)

6.  **配置 Docker MySQL 数据库**:
    * 确认 `docker-compose.yml` 文件在项目根目录 [cite: 100]。
    * **重要**: 修改 `docker-compose.yml` 文件中的 `MYSQL_ROOT_PASSWORD` 和 `MYSQL_PASSWORD` 为你自己的强密码 [cite: 3, 100]。
    * 在项目根目录的 WSL 终端中启动数据库服务:
        ```bash
        docker-compose up -d
        ```
        [cite: 3, 100]
    * 验证数据库容器是否运行: `docker ps` (应看到名为 `mysql_ecom_qa` 的容器状态为 Up) [cite: 3]。

7.  **配置 `.env` 文件**:
    * 在项目根目录创建 `.env` 文件 [cite: 4]。
    * 复制以下内容并 **替换为你自己的值**:
        ```dotenv
        # OpenAI API 配置
        OPENAI_API_KEY='sk-YOUR_OPENAI_API_KEY_HERE'
        # !!! 重要: 替换为你的 OpenAI 代理地址 !!!
        OPENAI_API_BASE='[https://your-openai-proxy-address.com/v1/](https://your-openai-proxy-address.com/v1/)' # 地址通常以 /v1 结尾

        # 数据库连接 URL (使用 docker-compose.yml 中设置的密码和端口 3307)
        # 格式: mysql+mysqlconnector://<user>:<password>@<host>:<port>/<database>
        # 如果密码包含特殊字符(@, :, / 等)，需要进行 URL 编码 (例如 @ 编码为 %40)
        DATABASE_URL='mysql+mysqlconnector://ecom_user:YOUR_URL_ENCODED_USER_PASSWORD@127.0.0.1:3307/ecom_qa_db' # [cite: 4, 101]

        # Flask 配置
        # !!! 重要: 生成一个长而随机的字符串作为密钥 !!!
        FLASK_SECRET_KEY='YOUR_VERY_STRONG_RANDOM_SECRET_KEY_HERE' # [cite: 4, 101]
        ```
    * **安全提示**: `.env` 文件包含敏感信息，已添加到 `.gitignore`，请勿提交到版本控制系统 [cite: 4, 7]。

8.  **创建数据库表**:
    * 确保 Docker MySQL 容器正在运行且 `.env` 文件配置正确。
    * 在激活了 `(ecom_qa)` 环境的项目根目录终端中运行:
        ```bash
        flask shell
        ```
    * 在打开的 Python Shell (提示符为 `>>>`) 中输入以下命令:
        ```python
        from app import db
        db.create_all()
        exit()
        ```
        [cite: 5, 6]
    * 如果 `db.create_all()` 没有报错，则表示表已成功创建。

## 5. 如何运行项目

1.  **启动数据库**:
    * 在项目根目录的 WSL 终端中:
        ```bash
        docker-compose up -d
        ```
        (如果数据库容器未运行)

2.  **激活 Conda 环境**:
    * 在项目根目录的 WSL 终端中:
        ```bash
        conda activate ecom_qa
        ```

3.  **启动 Flask 应用**:
    * 在激活了环境的项目根目录终端中:
        ```bash
        flask run
        ```
        或者使用 `run.py` 脚本 (效果相同):
        ```bash
        python run.py
        ```
        [cite: 5]
    * 应用默认运行在 `http://127.0.0.1:5000` 或 `http://localhost:5000` [cite: 5]。

4.  **访问应用**:
    * 在你的浏览器中打开 `http://localhost:5000`。

5.  **创建管理员账号 (首次运行)**:
    * 目前没有自动创建管理员的功能。你需要：
        1.  通过网站注册一个普通用户账号 (角色选 buyer 或 seller 均可) [cite: 28]。
        2.  使用数据库客户端 (如 DBeaver, VS Code 插件) 连接到 Docker MySQL (地址: `127.0.0.1`, 端口: `3307`, 用户名: `ecom_user`, 密码: 你在 `docker-compose.yml` 中设置的密码, 数据库: `ecom_qa_db`) [cite: 101]。
        3.  找到 `user` 表，将你刚刚注册的用户的 `role` 字段值手动修改为 `admin` [cite: 92]。
        4.  重新登录该账号，即可访问 `/admin` 后台。

## 6. 主要功能演示说明

1.  **注册/登录**:
    * 访问 `/auth/register` 注册买家和商家账号 [cite: 28]。
    * 访问 `/auth/login` 登录 [cite: 36]。
    * 导航栏根据登录状态和用户角色动态显示不同选项 [cite: 10]。
2.  **商品浏览**:
    * 首页 `/` 展示状态为 'active' 的商品列表 [cite: 36, 96]。
    * 点击商品链接进入 `/product/<id>` 查看详情 [cite: 39]。
3.  **商家中心**:
    * 使用商家账号登录后，点击导航栏 "商家中心" 进入 `/seller/dashboard` [cite: 1, 16]。
    * 可以添加、编辑、删除自己发布的商品 [cite: 7, 16, 22, 32]。
    * 仪表盘显示自己商品的列表和状态 [cite: 16, 96]。
    * 商家后台有专属的操作助手问答框。
4.  **智能问答**:
    * **通用问答**: 在网站底部固定的问答框提问通用问题 (如退货政策)，AI 会结合数据库中的 `FaqRule` 回答 [cite: 27, 39, 65, 92]。
    * **商品问答**: 在商品详情页 `/product/<id>` 的问答框提问关于该商品的问题 (如 "这个商品有什么颜色？")，AI 会结合商品信息和通用规则回答。
    * **商家操作问答**: 在商家后台 `/seller/dashboard` 的专属问答框提问后台操作问题 (如 "如何添加商品？")，AI 会优先使用 "商家操作" 类别的规则回答。
5.  **管理后台 (`/admin`)**:
    * 使用管理员账号登录后，点击导航栏 "管理后台" 下拉菜单访问。
    * **规则管理**: 查看、添加、编辑、删除 `FaqRule` 数据 [cite: 92, 94]。修改后问答系统使用的规则会实时更新 (缓存清除机制) [cite: 92]。
    * **用户管理**: 查看所有用户列表，查看用户详情，修改用户角色，重置用户密码，删除用户 (有安全检查) [cite: 96]。
    * **商品管理**: 查看所有商品列表 (含卖家和状态)，编辑任意商品信息，切换商品上下架状态 (`active`/`inactive`)，永久删除任意商品 [cite: 96, 97]。

## 7. 注意事项

* 请务必保护好你的 `.env` 文件中的 API Key 和数据库密码。
* OpenAI API 调用可能会产生费用，请注意用量。
* 本项目为原型系统，部分功能（如忘记密码、复杂商品规格、图片上传等）可能未完全实现或需要进一步完善。
* 管理员手动重置密码功能中的默认密码 (`Password123`) 非常不安全，仅供演示，真实场景需要更安全的机制 [cite: 96]。


