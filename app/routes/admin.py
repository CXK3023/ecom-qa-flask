# app/routes/admin.py
# 确保导入了所有需要的东西
from flask import Blueprint, render_template, abort, flash, redirect, url_for, request
from flask_login import login_required, current_user
# 导入所有需要用到的模型
from ..models import FaqRule, User, Product
from .. import db
from functools import wraps # 用于自定义装饰器
from ..services.data_service import clear_rules_cache # 导入清除缓存函数
# 可能需要导入这个来处理密码哈希 (虽然我们在模型里用了 set_password)
from werkzeug.security import generate_password_hash

# 创建一个名为 'admin' 的蓝图
admin = Blueprint('admin', __name__)

# --- 权限控制：自定义 admin_required 装饰器 (保持不变) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or getattr(current_user, 'role', None) != 'admin':
            flash('您必须以管理员身份登录才能访问此页面。', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES FOR RULE MANAGEMENT (保持不变) ---
@admin.route('/rules')
@login_required
@admin_required
def list_rules():
    """显示所有规则/FAQ 列表"""
    try:
        all_rules = FaqRule.query.order_by(FaqRule.category, FaqRule.id).all()
    except Exception as e:
        flash(f'加载规则列表时出错: {e}', 'error')
        all_rules = []
    return render_template('admin/rules_list.html', rules=all_rules)

@admin.route('/rules/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_rule():
    """显示添加规则表单 (GET) 或处理表单提交 (POST)"""
    if request.method == 'POST':
        category = request.form.get('category')
        question = request.form.get('question')
        answer = request.form.get('answer')
        error = None
        if not category: error = '类别不能为空。'
        elif not question: error = '问题不能为空。'
        elif not answer: error = '答案不能为空。'

        if error is None:
            try:
                new_rule = FaqRule(category=category, question=question, answer=answer)
                db.session.add(new_rule)
                db.session.commit()
                clear_rules_cache() # 清除缓存
                flash('新规则添加成功！', 'success')
                return redirect(url_for('admin.list_rules'))
            except Exception as e:
                db.session.rollback()
                flash(f'添加规则时发生错误: {e}', 'error')
        else:
            flash(error, 'error')
    return render_template('admin/add_edit_rule.html', form_data={}, action="add")

@admin.route('/rules/edit/<int:rule_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_rule(rule_id):
    """显示编辑规则表单 (GET) 或处理更新 (POST)"""
    rule_to_edit = FaqRule.query.get_or_404(rule_id)
    if request.method == 'POST':
        category = request.form.get('category')
        question = request.form.get('question')
        answer = request.form.get('answer')
        error = None
        if not category: error = '类别不能为空。'
        elif not question: error = '问题不能为空。'
        elif not answer: error = '答案不能为空。'

        if error is None:
            try:
                rule_to_edit.category = category
                rule_to_edit.question = question
                rule_to_edit.answer = answer
                db.session.commit()
                clear_rules_cache() # 清除缓存
                flash('规则更新成功！', 'success')
                return redirect(url_for('admin.list_rules'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新规则时发生错误: {e}', 'error')
        else:
            flash(error, 'error')

    form_data = {
        'id': rule_to_edit.id,
        'category': rule_to_edit.category,
        'question': rule_to_edit.question,
        'answer': rule_to_edit.answer
    }
    return render_template('admin/add_edit_rule.html', form_data=form_data, action="edit")

@admin.route('/rules/delete/<int:rule_id>', methods=['POST'])
@login_required
@admin_required
def delete_rule(rule_id):
    """处理删除规则的请求"""
    rule_to_delete = FaqRule.query.get_or_404(rule_id)
    try:
        db.session.delete(rule_to_delete)
        db.session.commit()
        clear_rules_cache() # 清除缓存
        flash(f'规则 ID: {rule_id} 已成功删除！', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除规则时发生错误: {e}', 'error')
    return redirect(url_for('admin.list_rules'))


# --- ROUTES FOR USER MANAGEMENT ---
@admin.route('/users')
@login_required
@admin_required
def list_users():
    """Display a list of all registered users."""
    try:
        all_users = User.query.order_by(User.id).all()
        print(f"DEBUG: Found {len(all_users)} users.") # 保留调试日志
    except Exception as e:
        flash(f'加载用户列表时出错: {e}', 'error')
        print(f"ERROR: Loading users failed: {e}") # 保留错误日志
        all_users = []
    return render_template('admin/users_list.html', users=all_users)

@admin.route('/users/view/<int:user_id>')
@login_required
@admin_required
def view_user(user_id):
    """Display details for a specific user."""
    user = User.query.get_or_404(user_id)
    user_products = []
    if user.role == 'seller':
        try:
            user_products = Product.query.filter_by(seller_id=user.id).order_by(Product.id.desc()).all()
            print(f"DEBUG: Found {len(user_products)} products for seller {user.username}") # 保留调试日志
        except Exception as e:
            flash(f'加载卖家商品列表时出错: {e}', 'error')
            print(f"ERROR: Loading seller products failed for user {user_id}: {e}") # 保留错误日志
    return render_template('admin/user_view.html', user=user, products=user_products)

# --- vvvvvv 这是新添加的函数 vvvvvv ---
@admin.route('/users/edit/<int:user_id>', methods=['GET', 'POST']) # 路径包含用户ID，允许GET和POST
@login_required
@admin_required
def edit_user(user_id):
    """Edit user role."""
    # 根据 ID 查找要编辑的用户，找不到就 404
    user_to_edit = User.query.get_or_404(user_id)

    # --- 处理用户提交表单的情况 (POST 请求) ---
    if request.method == 'POST':
        # 从提交的表单里获取用户选择的新角色
        new_role = request.form.get('role')
        # 定义允许的角色有哪些
        allowed_roles = ['buyer', 'seller', 'admin']

        if new_role in allowed_roles: # 检查选择的角色是否有效
            # 增加一个检查，防止把最后一个admin改成非admin
            if user_to_edit.role == 'admin' and new_role != 'admin':
                admin_count = User.query.filter_by(role='admin').count()
                if admin_count <= 1:
                    flash('不能将最后一个管理员的角色更改为非管理员！', 'error')
                    # 仍然渲染编辑页面，但不进行更改
                    return render_template('admin/edit_user.html', user=user_to_edit)

            try:
                # 把数据库里这个用户的角色更新为新选择的角色
                user_to_edit.role = new_role
                db.session.commit() # 保存更改到数据库
                flash(f'用户 {user_to_edit.username} 的角色已成功更新为 {new_role}！', 'success') # 显示成功提示
                # 操作成功后，跳转回用户列表页面
                return redirect(url_for('admin.list_users'))
            except Exception as e:
                # 如果保存数据库时出错
                db.session.rollback() # 撤销更改
                flash(f'更新用户角色时发生错误: {e}', 'error') # 显示错误提示
                print(f"ERROR: Updating role failed for user {user_id}: {e}") # 打印错误日志
        else:
            # 如果用户选的角色无效
            flash('选择的角色无效！', 'error')

        # 注意：如果 POST 请求处理失败 (比如数据库保存出错或角色无效)，
        # 代码会继续往下走，重新显示编辑表单，并带上错误提示。

    # --- 处理第一次访问编辑页面 (GET 请求) ---
    # 或者 POST 请求失败后，也会执行到这里
    # 渲染编辑用户的 HTML 页面 (我们下一步创建它)
    # 把要编辑的用户对象 user_to_edit 传给模板，这样模板能显示用户的当前信息
    return render_template('admin/edit_user.html', user=user_to_edit)
# --- ^^^^^^ 新添加的函数结束 ^^^^^^ ---


# app/routes/admin.py (继续添加代码)

# 确保文件顶部导入了 User 模型和 db
# 应该已经导入了 Flask 相关函数 flash, redirect, url_for

@admin.route('/users/reset_password/<int:user_id>', methods=['POST']) # 只允许 POST 请求
@login_required
@admin_required
def reset_user_password(user_id):
    """Reset user's password to a default value."""
    # 查找要重置密码的用户，找不到就 404
    user_to_reset = User.query.get_or_404(user_id)

    # 定义一个简单的临时密码 (!! 注意：在真实项目中，密码不应该这么简单 !!)
    temp_password = "Password123" 

    try:
        # 使用 User 模型里自带的 set_password 方法来设置密码，它会自动加密
        user_to_reset.set_password(temp_password) 
        db.session.commit() # 保存密码更改到数据库
        # 显示一个警告消息，告诉管理员临时密码是什么，并提醒他通知用户
        flash(f'用户 {user_to_reset.username} 的密码已成功重置为: {temp_password} (请务必告知用户并建议其尽快修改！)', 'warning') 
    except Exception as e:
        # 如果保存数据库出错
        db.session.rollback() # 撤销更改
        flash(f'重置密码时发生错误: {e}', 'error') # 显示错误提示
        print(f"ERROR: Resetting password failed for user {user_id}: {e}") # 打印错误日志

    # 不管成功还是失败，都重定向回用户列表页面
    return redirect(url_for('admin.list_users'))



@admin.route('/users/delete/<int:user_id>', methods=['POST']) # 只允许 POST 请求
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user account, with safety checks."""

    # --- 安全检查 1：不能删除自己 ---
    if user_id == current_user.id:
         flash('操作无效：不能删除自己的管理员账户！', 'error')
         return redirect(url_for('admin.list_users'))

    # --- 查找要删除的用户 ---
    user_to_delete = User.query.get_or_404(user_id)

    # --- 安全检查 2：如果用户是卖家，检查是否有商品 ---
    # (对于买家或没有商品的卖家，product_count 会是 0)
    try:
        product_count = Product.query.filter_by(seller_id=user_id).count()
        if product_count > 0:
            flash(f'删除失败：用户 {user_to_delete.username} 仍有 {product_count} 件关联商品。请先将这些商品删除或转移给其他卖家。', 'error')
            return redirect(url_for('admin.list_users'))
    except Exception as e:
         flash(f'检查关联商品时出错: {e}', 'error')
         print(f"ERROR: Checking products failed for user {user_id} before delete: {e}")
         return redirect(url_for('admin.list_users'))
    # --- 检查结束 ---

    # --- 执行删除操作 ---
    try:
        username = user_to_delete.username # 记录用户名，用于提示
        db.session.delete(user_to_delete) # 从数据库会话中标记删除
        db.session.commit() # 确认删除
        flash(f'用户 {username} (ID: {user_id}) 已被成功删除。', 'success') # 显示成功消息
    except Exception as e:
        db.session.rollback() # 如果删除时出错，撤销操作
        flash(f'删除用户时发生错误: {e}', 'error') # 显示错误消息
        print(f"ERROR: Deleting user {user_id} failed: {e}") # 打印错误日志

    # 重定向回用户列表页面
    return redirect(url_for('admin.list_users'))



# --- ROUTES FOR PRODUCT MANAGEMENT (ADMIN) ---
@admin.route('/products') # 管理员查看所有商品的路径 /admin/products
@login_required
@admin_required
def list_all_products():
    """Display a list of all products from all sellers."""
    try:
        # --- 查询数据库，获取所有商品以及对应的卖家用户名 ---
        # 我们需要联合查询 Product 表和 User 表
        # db.session.query(Product, User.username) 表示同时选择 Product 对象和 User 的 username
        # .join(User, Product.seller_id == User.id) 表示通过 Product.seller_id 和 User.id 把两张表连起来
        # .order_by(...) 可以按状态或ID排序
        products_with_sellers = db.session.query(Product, User.username)\
                                 .join(User, Product.seller_id == User.id)\
                                 .order_by(Product.status, Product.id.desc())\
                                 .all()
        # 查询结果 'products_with_sellers' 会是一个列表，
        # 列表里的每一项都是一个元组(tuple)，包含 (Product对象, 卖家用户名字符串)
        # 例如：[(<Product A>, 'seller1'), (<Product B>, 'seller2')]
        print(f"DEBUG: Found {len(products_with_sellers)} total products.")
    except Exception as e:
        flash(f'加载所有商品列表时出错: {e}', 'error')
        print(f"ERROR: Loading all products failed: {e}")
        products_with_sellers = []

    # 渲染一个新的 HTML 模板 products_list.html，把查询结果传过去
    return render_template('admin/products_list.html', products_with_sellers=products_with_sellers)

# app/routes/admin.py (继续添加代码)

# 确保文件顶部导入了 Product 模型, db, request, flash, redirect, url_for 等

@admin.route('/products/edit/<int:product_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_product(product_id):
    """Allow admin to edit any product's details."""
    # 查找要编辑的商品，找不到就 404
    product_to_edit = Product.query.get_or_404(product_id)

    # --- 处理表单提交 (POST 请求) ---
    if request.method == 'POST':
        # 从表单获取数据
        name = request.form.get('name')
        description = request.form.get('description')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        # 注意：管理员编辑时通常不改变卖家，所以不处理 seller_id

        # --- 数据验证 (与商家添加/编辑商品时类似) ---
        error = None
        price = None
        stock = None

        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'

        if price_str:
            try:
                price = float(price_str)
                if price < 0: error = '价格不能为负数。'
            except ValueError:
                error = '价格必须是有效的数字。'

        if stock_str:
            try:
                stock = int(stock_str)
                if stock < 0: error = '库存不能为负数。'
            except ValueError:
                error = '库存必须是有效的整数。'
        # --- 验证结束 ---

        if error is None:
            # 如果数据有效，更新商品对象的属性
            try:
                product_to_edit.name = name
                product_to_edit.description = description
                product_to_edit.price = price
                product_to_edit.stock = stock
                # 管理员编辑时，我们通常保持 status 不变，除非有单独的状态管理功能
                # product_to_edit.status = request.form.get('status') # 如果需要在这里改状态

                db.session.commit() # 保存更改
                flash(f'商品 "{product_to_edit.name}" (ID: {product_id}) 更新成功！', 'success')
                # 成功后跳转回管理员的商品列表页面
                return redirect(url_for('admin.list_all_products'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生错误: {e}', 'error')
                print(f"ERROR: Updating product {product_id} failed: {e}")
        else:
            # 如果验证失败，显示错误
            flash(error, 'error')

        # 如果 POST 处理失败 (验证错误或数据库错误)，会继续向下执行，重新显示编辑表单

    # --- 显示编辑表单 (GET 请求 或 POST 失败) ---
    # 渲染编辑商品的 HTML 模板 (我们马上创建它)
    # 把商品对象传递给模板，用于预填充表单
    return render_template('admin/edit_product.html', product=product_to_edit)


# app/routes/admin.py (继续添加代码)

# 确保文件顶部导入了 Product, db, flash, redirect, url_for

@admin.route('/products/toggle_status/<int:product_id>', methods=['POST']) # 只允许 POST 请求
@login_required
@admin_required
def toggle_product_status(product_id):
    """Toggle product status between active and inactive."""
    # 查找要切换状态的商品，找不到就 404
    product_to_toggle = Product.query.get_or_404(product_id)

    try:
        # 检查当前状态并切换
        if product_to_toggle.status == 'active':
            product_to_toggle.status = 'inactive' # 如果是 active，就改成 inactive
            action_text = "下架"
        else:
            product_to_toggle.status = 'active'   # 否则 (是 inactive 或其他)，就改成 active
            action_text = "上架"

        db.session.commit() # 保存状态更改到数据库
        flash(f'商品 "{product_to_toggle.name}" 已成功{action_text}！', 'success') # 显示成功消息
    except Exception as e:
        db.session.rollback() # 如果出错，撤销更改
        flash(f'切换商品状态时发生错误: {e}', 'error') # 显示错误消息
        print(f"ERROR: Toggling status failed for product {product_id}: {e}") # 打印错误日志

    # 不管成功还是失败，都重定向回管理员的商品列表页面
    return redirect(url_for('admin.list_all_products'))


# app/routes/admin.py (继续添加代码)

# 确保文件顶部导入了 Product, db, flash, redirect, url_for

@admin.route('/products/delete/<int:product_id>', methods=['POST']) # 只允许 POST 请求
@login_required
@admin_required
def delete_product(product_id):
    """Permanently delete a product."""
    # 查找要删除的商品，找不到就 404
    product_to_delete = Product.query.get_or_404(product_id)

    try:
        product_name = product_to_delete.name # 记录商品名称，用于提示
        db.session.delete(product_to_delete) # 从数据库会话中标记删除
        db.session.commit() # 确认删除
        flash(f'商品 "{product_name}" (ID: {product_id}) 已被永久删除。', 'success') # 显示成功消息
    except Exception as e:
        db.session.rollback() # 如果出错，撤销操作
        flash(f'删除商品时发生错误: {e}', 'error') # 显示错误消息
        print(f"ERROR: Deleting product {product_id} failed: {e}") # 打印错误日志

    # 不管成功还是失败，都重定向回管理员的商品列表页面
    return redirect(url_for('admin.list_all_products'))

