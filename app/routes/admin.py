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

# --- 权限控制：自定义 admin_required 装饰器 (修改版) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # === vvv MODIFY START vvv ===
        # 检查：用户必须已登录，并且 (用户的角色是 'admin' 或者 用户的 ID 是 1)
        # 如果不满足这些条件，则拒绝访问
        if not current_user.is_authenticated or not (current_user.role == 'admin' or current_user.id == 1):
        # === ^^^ MODIFY END ^^^ ===
            flash('您必须以管理员身份登录才能访问此页面。', 'error')
            return redirect(url_for('main.index')) # 或者跳转到登录页 url_for('auth.login')
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES FOR RULE MANAGEMENT ---
@admin.route('/rules')
@login_required
@admin_required # 应用修改后的装饰器
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
@admin_required # 应用修改后的装饰器
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
@admin_required # 应用修改后的装饰器
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
@admin_required # 应用修改后的装饰器
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
@admin_required # 应用修改后的装饰器
def list_users():
    """Display a list of all registered users."""
    try:
        all_users = User.query.order_by(User.id).all()
        print(f"DEBUG: Found {len(all_users)} users.")
    except Exception as e:
        flash(f'加载用户列表时出错: {e}', 'error')
        print(f"ERROR: Loading users failed: {e}")
        all_users = []
    return render_template('admin/users_list.html', users=all_users)

@admin.route('/users/view/<int:user_id>')
@login_required
@admin_required # 应用修改后的装饰器
def view_user(user_id):
    """Display details for a specific user."""
    user = User.query.get_or_404(user_id)
    user_products = []
    if user.role == 'seller':
        try:
            user_products = Product.query.filter_by(seller_id=user.id).order_by(Product.id.desc()).all()
            print(f"DEBUG: Found {len(user_products)} products for seller {user.username}")
        except Exception as e:
            flash(f'加载卖家商品列表时出错: {e}', 'error')
            print(f"ERROR: Loading seller products failed for user {user_id}: {e}")
    return render_template('admin/user_view.html', user=user, products=user_products)

@admin.route('/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required # 应用修改后的装饰器
def edit_user(user_id):
    """Edit user role."""
    user_to_edit = User.query.get_or_404(user_id)

    if request.method == 'POST':
        new_role = request.form.get('role')
        allowed_roles = ['buyer', 'seller', 'admin']

        if new_role in allowed_roles:
             # 安全检查：不能将最后一个管理员（无论是通过角色还是 ID=1 识别的）降级
             is_last_admin = False
             if (user_to_edit.role == 'admin' or user_to_edit.id == 1) and new_role != 'admin':
                 # 统计数据库中角色为 'admin' 的用户数
                 admin_role_count = User.query.filter_by(role='admin').count()
                 # 如果当前用户是 ID=1 且数据库角色不是 admin，则实际 admin 数为 admin_role_count
                 # 如果当前用户是 ID=1 且数据库角色是 admin，则实际 admin 数为 admin_role_count
                 # 如果当前用户不是 ID=1 但数据库角色是 admin，则实际 admin 数为 admin_role_count
                 # 我们需要考虑 ID=1 这个特殊用户，即使他数据库角色不是admin，也算一个管理员
                 effective_admin_count = admin_role_count
                 user_1 = User.query.get(1)
                 # 如果 ID=1 的用户存在且其角色不是 admin，那么有效管理员数量要加 1
                 if user_1 and user_1.role != 'admin':
                     effective_admin_count += 1

                 # 如果有效管理员数量 <= 1，则不允许降级
                 if effective_admin_count <= 1:
                     is_last_admin = True
                     flash('操作失败：不能将系统中唯一的管理员权限移除！', 'error')
                     # 保持在编辑页面
                     return render_template('admin/edit_user.html', user=user_to_edit)

             if not is_last_admin:
                try:
                    user_to_edit.role = new_role
                    db.session.commit()
                    flash(f'用户 {user_to_edit.username} 的角色已成功更新为 {new_role}！', 'success')
                    return redirect(url_for('admin.list_users'))
                except Exception as e:
                    db.session.rollback()
                    flash(f'更新用户角色时发生错误: {e}', 'error')
                    print(f"ERROR: Updating role failed for user {user_id}: {e}")
        else:
            flash('选择的角色无效！', 'error')

    return render_template('admin/edit_user.html', user=user_to_edit)


@admin.route('/users/reset_password/<int:user_id>', methods=['POST'])
@login_required
@admin_required # 应用修改后的装饰器
def reset_user_password(user_id):
    """Reset user's password to a default value."""
    user_to_reset = User.query.get_or_404(user_id)
    temp_password = "Password123"

    try:
        user_to_reset.set_password(temp_password)
        db.session.commit()
        flash(f'用户 {user_to_reset.username} 的密码已成功重置为: {temp_password} (请务必告知用户并建议其尽快修改！)', 'warning')
    except Exception as e:
        db.session.rollback()
        flash(f'重置密码时发生错误: {e}', 'error')
        print(f"ERROR: Resetting password failed for user {user_id}: {e}")

    return redirect(url_for('admin.list_users'))

@admin.route('/users/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required # 应用修改后的装饰器
def delete_user(user_id):
    """Delete a user account, with safety checks."""

    # 安全检查 1：不能删除自己 (无论是否为 ID=1)
    if user_id == current_user.id:
         flash('操作无效：不能删除自己的账户！', 'error')
         return redirect(url_for('admin.list_users'))

    # 安全检查：不能删除 ID=1 的用户 (超级管理员)
    if user_id == 1:
        flash('操作无效：不能删除 ID 为 1 的超级管理员账户！', 'error')
        return redirect(url_for('admin.list_users'))

    user_to_delete = User.query.get_or_404(user_id)

    # 安全检查 2：如果用户是卖家，检查是否有商品
    try:
        product_count = Product.query.filter_by(seller_id=user_id).count()
        if product_count > 0:
            flash(f'删除失败：用户 {user_to_delete.username} 仍有 {product_count} 件关联商品。请先处理这些商品。', 'error')
            return redirect(url_for('admin.list_users'))
    except Exception as e:
         flash(f'检查关联商品时出错: {e}', 'error')
         print(f"ERROR: Checking products failed for user {user_id} before delete: {e}")
         return redirect(url_for('admin.list_users'))

    # 执行删除操作
    try:
        username = user_to_delete.username
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f'用户 {username} (ID: {user_id}) 已被成功删除。', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除用户时发生错误: {e}', 'error')
        print(f"ERROR: Deleting user {user_id} failed: {e}")

    return redirect(url_for('admin.list_users'))



# --- ROUTES FOR PRODUCT MANAGEMENT (ADMIN) ---
@admin.route('/products')
@login_required
@admin_required # 应用修改后的装饰器
def list_all_products():
    """Display a list of all products from all sellers."""
    try:
        products_with_sellers = db.session.query(Product, User.username)\
                                 .join(User, Product.seller_id == User.id)\
                                 .order_by(Product.status, Product.id.desc())\
                                 .all()
        print(f"DEBUG: Found {len(products_with_sellers)} total products.")
    except Exception as e:
        flash(f'加载所有商品列表时出错: {e}', 'error')
        print(f"ERROR: Loading all products failed: {e}")
        products_with_sellers = []

    return render_template('admin/products_list.html', products_with_sellers=products_with_sellers)

@admin.route('/products/edit/<int:product_id>', methods=['GET', 'POST'])
@login_required
@admin_required # 应用修改后的装饰器
def edit_product(product_id):
    """Allow admin to edit any product's details."""
    product_to_edit = Product.query.get_or_404(product_id)

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
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

        if error is None:
            try:
                product_to_edit.name = name
                product_to_edit.description = description
                product_to_edit.price = price
                product_to_edit.stock = stock
                db.session.commit()
                flash(f'商品 "{product_to_edit.name}" (ID: {product_id}) 更新成功！', 'success')
                return redirect(url_for('admin.list_all_products'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生错误: {e}', 'error')
                print(f"ERROR: Updating product {product_id} failed: {e}")
        else:
            flash(error, 'error')

    return render_template('admin/edit_product.html', product=product_to_edit)


@admin.route('/products/toggle_status/<int:product_id>', methods=['POST'])
@login_required
@admin_required # 应用修改后的装饰器
def toggle_product_status(product_id):
    """Toggle product status between active and inactive."""
    product_to_toggle = Product.query.get_or_404(product_id)

    try:
        if product_to_toggle.status == 'active':
            product_to_toggle.status = 'inactive'
            action_text = "下架"
        else:
            product_to_toggle.status = 'active'
            action_text = "上架"

        db.session.commit()
        flash(f'商品 "{product_to_toggle.name}" 已成功{action_text}！', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'切换商品状态时发生错误: {e}', 'error')
        print(f"ERROR: Toggling status failed for product {product_id}: {e}")

    return redirect(url_for('admin.list_all_products'))


@admin.route('/products/delete/<int:product_id>', methods=['POST'])
@login_required
@admin_required # 应用修改后的装饰器
def delete_product(product_id):
    """Permanently delete a product."""
    product_to_delete = Product.query.get_or_404(product_id)

    try:
        product_name = product_to_delete.name
        db.session.delete(product_to_delete)
        db.session.commit()
        flash(f'商品 "{product_name}" (ID: {product_id}) 已被永久删除。', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除商品时发生错误: {e}', 'error')
        print(f"ERROR: Deleting product {product_id} failed: {e}")

    return redirect(url_for('admin.list_all_products'))