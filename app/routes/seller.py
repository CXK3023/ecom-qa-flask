# app/routes/seller.py
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort # [source: 1]
from flask_login import login_required, current_user # [source: 1]
from ..models import Product, User # 导入 Product 和 User 模型 # [source: 1]
from .. import db # [source: 1]
from functools import wraps # 导入 wraps，用于创建我们自己的装饰器 # [source: 1]

# 创建一个名叫 'seller' 的蓝图
seller = Blueprint('seller', __name__) # [source: 1]

# --- 自定义装饰器：检查用户是否为商家 ---
def seller_required(f): # [source: 1]
    """检查当前用户是否是商家角色"""
    @wraps(f) # [source: 1]
    def decorated_function(*args, **kwargs): # [source: 1]
        if not current_user.is_authenticated or current_user.role != 'seller': # [source: 1]
            # 如果用户未登录，或者登录了但不是 'seller' 角色
            flash('您需要以商家身份登录才能访问此页面。', 'error') # [source: 2]
            abort(403) # 返回 403 Forbidden 错误 # [source: 2]
            # 或者可以重定向到登录页: return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs) # [source: 2]
    return decorated_function # [source: 2]

# --- 商家仪表盘路由 ---
@seller.route('/dashboard') # [source: 2]
@login_required # 必须先登录 # [source: 2]
@seller_required # 必须是商家 # [source: 2]
def dashboard():
    """显示商家仪表盘，列出自己的商品""" # [source: 2]
    # 查询当前商家名下的所有商品
    products = Product.query.filter_by(seller_id=current_user.id).order_by(Product.id.desc()).all() # [source: 3]
    # 稍后我们会添加查询逻辑，现在先渲染模板
    return render_template('seller/dashboard.html', products=products) # 把查询结果传递给模板

# --- 后面会在这里添加 添加/编辑/删除 商品的路由 ---
# app/routes/seller.py (在文件末尾添加)
@seller.route('/add', methods=['GET', 'POST']) # [source: 7]
@login_required # [source: 7]
@seller_required # [source: 7]
def add_product():
    """处理添加新商品""" # [source: 7]
    if request.method == 'POST': # [source: 7]
        name = request.form.get('name') # [source: 7]
        description = request.form.get('description') # [source: 7]
        price_str = request.form.get('price') # [source: 7]
        stock_str = request.form.get('stock') # [source: 8]
        error = None # [source: 8]

        if not name: # [source: 8]
            error = '商品名称不能为空。' # [source: 8]
        if not price_str: # [source: 8]
            error = '价格不能为空。' # [source: 8]
        if not stock_str: # [source: 8]
            error = '库存不能为空。' # [source: 8]

        price = None # [source: 8]
        stock = None # [source: 8]

        if price_str: # [source: 9]
            try:
                price = float(price_str) # 尝试将价格转换为数字(可以有小数) # [source: 9]
                if price < 0: error = '价格不能为负数。' # [source: 9]
            except ValueError: # [source: 9]
                error = '价格必须是有效的数字。' # [source: 9]

        if stock_str: # [source: 10]
            try:
                stock = int(stock_str) # 尝试将库存转换为整数 # [source: 10]
                if stock < 0: error = '库存不能为负数。' # [source: 10]
            except ValueError: # [source: 10]
                error = '库存必须是有效的整数。' # [source: 10]

        if error is None: # [source: 11]
            # 创建新商品对象，关联当前商家
            new_product = Product(name=name, description=description, price=price, stock=stock, seller_id=current_user.id) # [source: 11]
            try: # [source: 11]
                db.session.add(new_product) # [source: 11]
                db.session.commit() # [source: 11]
                flash('商品添加成功！', 'success') # [source: 11]
                return redirect(url_for('seller.dashboard')) # 添加成功后返回仪表盘 # [source: 12]
            except Exception as e: # [source: 12]
                db.session.rollback() # [source: 12]
                flash(f'添加商品时出错: {e}', 'error') # [source: 12]
        else:
            flash(error, 'error') # [source: 12]

    # 如果是 GET 请求或 POST 出错，显示添加表单
    return render_template('seller/add_product.html') # [source: 13]


# app/routes/seller.py (在文件末尾添加)
# from flask import abort # 再次确认这行在文件顶部导入了

@seller.route('/edit/<int:product_id>', methods=['GET', 'POST']) # [source: 22]
@login_required # [source: 22]
@seller_required # [source: 22]
def edit_product(product_id):
    """处理编辑商品信息""" # [source: 22]
    # 使用 get_or_404 查询商品，如果找不到对应的商品 ID，会自动返回 404 页面
    product = Product.query.get_or_404(product_id) # [source: 22]

    # !!! 非常重要的安全检查：确保当前登录的用户就是这个商品的主人 !!!
    if product.seller_id != current_user.id: # [source: 23]
        # 如果商品记录的卖家 ID 不等于 当前登录用户的 ID
        flash('您没有权限编辑这个商品。', 'error') # 准备错误消息 # [source: 23]
        abort(403) # 立刻停止处理，并向浏览器返回 403 Forbidden (禁止访问) 错误 # [source: 23]

    # 如果是用户提交了编辑表单 (POST 请求)
    if request.method == 'POST': # [source: 23]
        # 从提交的表单里获取用户输入的新信息
        name = request.form.get('name') # [source: 23]
        description = request.form.get('description') # [source: 23]
        price_str = request.form.get('price') # [source: 23]
        stock_str = request.form.get('stock') # [source: 23]

        # --- 数据验证 ---
        # （这里的验证逻辑和 add_product 函数里的几乎一样，
        #    检查名称、价格、库存是否为空、是否为有效数字、是否为负数等。
        #    你可以直接从 add_product 函数复制验证部分的代码过来替换掉下面的注释）
        error = None # [source: 24]
        # --- 在这里添加和 add_product 类似的验证代码 ---
        if not name: error = '商品名称不能为空。' # [source: 24]
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        price = None # [source: 24]
        stock = None # [source: 24]
        if price_str: # [source: 24]
            try: # [source: 24]
                price = float(price_str) # [source: 25]
                if price < 0: error = '价格不能为负数。' # [source: 25]
            except ValueError: error = '价格必须是有效的数字。' # [source: 25]
        if stock_str: # [source: 25]
            try: # [source: 25]
                stock = int(stock_str) # [source: 26]
                if stock < 0: error = '库存不能为负数。' # [source: 26]
            except ValueError: error = '库存必须是有效的整数。' # [source: 26]
        # --- 验证结束 ---

        # 如果验证通过，没有错误
        if error is None: # [source: 26]
            # !!! 更新商品对象的属性 !!!
            # 注意：这里不是创建新对象，而是修改我们从数据库读出来的那个 product 对象的属性
            product.name = name # [source: 26]
            product.description = description # [source: 27]
            product.price = price # [source: 27]
            product.stock = stock # [source: 27]
            try: # 尝试保存更改 # [source: 27]
                # 因为 product 是从数据库会话中读出来的，我们修改了它之后，
                # 直接调用 commit() 就可以把更改保存回数据库了。
                db.session.commit() # [source: 27]
                flash('商品信息更新成功！', 'success') # [source: 27]
                return redirect(url_for('seller.dashboard')) # 跳转回仪表盘 # [source: 28]
            except Exception as e: # [source: 28]
                db.session.rollback() # 保存出错，撤销更改 # [source: 28]
                flash(f'更新商品时出错: {e}', 'error') # [source: 28]
        else:
            # 如果验证有错，准备错误消息
            flash(error, 'error') # [source: 28]
        # 注意：即时 POST 请求处理中有错误 (比如验证失败)，我们也不能直接返回模板。
        # 因为如果验证失败，我们需要让用户留在编辑页面看到错误提示，
        # 并且页面需要能访问到 product 对象来重新填充表单 (虽然这次填充的是用户提交的错误数据)。
        # 所以，处理 POST 请求的末尾，如果验证失败，不能直接 return，而是让代码继续往下走。

    # 如果是用户第一次访问编辑页面 (GET 请求)，或者 POST 请求处理中有错误
    # 就显示编辑商品的网页表单 (我们下一步创建它)
    # **非常重要**：我们需要把从数据库查到的 product 对象传递给模板，
    # 这样模板才能把商品原来的信息预先填入表单。
    return render_template('seller/edit_product.html', product=product) # [source: 29]



# app/routes/seller.py (在文件末尾添加)
@seller.route('/delete/<int:product_id>', methods=['POST']) # 只允许 POST 请求来触发删除 # [source: 32]
@login_required # [source: 32]
@seller_required # [source: 32]
def delete_product(product_id):
    """处理删除商品""" # [source: 32]
    # 还是先尝试获取商品，找不到就 404
    product = Product.query.get_or_404(product_id) # [source: 33]

    # !!! 再次进行所有权安全检查 !!! (和编辑时一样重要)
    if product.seller_id != current_user.id: # [source: 34]
        flash('您没有权限删除这个商品。', 'error') # [source: 34]
        abort(403) # [source: 34]

    # 如果权限检查通过，尝试删除
    try: # [source: 34]
        db.session.delete(product) # 告诉数据库会话：准备删除这个 product 对象 # [source: 34]
        db.session.commit() # 确认执行删除操作 # [source: 34]
        flash('商品删除成功！', 'success') # [source: 34]
    except Exception as e: # [source: 34]
        db.session.rollback() # 如果删除过程中出错，撤销操作 # [source: 34]
        flash(f'删除商品时出错: {e}', 'error') # [source: 34]

    # 无论成功还是失败，都跳转回商家的仪表盘页面
    return redirect(url_for('seller.dashboard')) # [source: 34]
