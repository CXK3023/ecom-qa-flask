# app/routes/admin.py
# === vvv IMPORTS vvv ===
from flask import Blueprint, render_template, abort, flash, redirect, url_for, request, current_app, Response # 添加 Response
from flask_login import login_required, current_user
from ..models import FaqRule, User, Product
from .. import db
from functools import wraps
from ..services.data_service import clear_rules_cache
from werkzeug.security import generate_password_hash
import os
from werkzeug.utils import secure_filename
import csv
import io
from datetime import datetime
# === ^^^ IMPORTS ^^^ ===

admin = Blueprint('admin', __name__)

# === vvv HELPER FUNCTIONS (Image Handling) vvv ===
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    # 检查 ALLOWED_EXTENSIONS 是否存在且不为空
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS')
    if not allowed_extensions:
        print("WARN: ALLOWED_EXTENSIONS not configured in Flask app.")
        return False # 如果未配置，则不允许任何文件
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_product_image(file, current_image_filename=None):
    """处理上传的图片文件：验证、保存、删除旧文件。返回保存后的文件名或 None。"""
    if not file or file.filename == '': return None
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 实践中建议重命名以防冲突:
        # filename = f"{uuid.uuid4()}_{filename}"
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        if not upload_folder:
             flash("错误：文件上传路径未配置。", "error")
             print("ERROR: UPLOAD_FOLDER not configured.")
             return None
        save_path = os.path.join(upload_folder, filename)

        # 删除旧图 (如果需要且存在)
        if current_image_filename and current_image_filename != filename:
            old_image_path = os.path.join(upload_folder, current_image_filename)
            if os.path.exists(old_image_path):
                try:
                    os.remove(old_image_path)
                    print(f"DEBUG: Admin deleted old image: {old_image_path}")
                except Exception as e:
                    print(f"ERROR deleting old image {old_image_path}: {e}")
                    flash(f"删除旧图片 '{current_image_filename}' 失败: {e}", 'warning')
            else:
                 print(f"WARN: Admin found old image path not existing: {old_image_path}")
        # 保存新图
        try:
            os.makedirs(upload_folder, exist_ok=True)
            file.save(save_path)
            print(f"DEBUG: Admin saved image to {save_path}")
            return filename
        except Exception as e:
            flash(f"图片保存失败: {e}", "error")
            print(f"ERROR saving image to {save_path}: {e}")
            return None
    else:
        allowed_formats = ", ".join(current_app.config.get('ALLOWED_EXTENSIONS', set()))
        flash(f'上传失败：只允许上传 {allowed_formats} 格式的图片。', 'error')
        return None

def delete_product_image(image_filename):
    """删除指定的商品图片文件"""
    if not image_filename: return False
    upload_folder = current_app.config.get('UPLOAD_FOLDER')
    if not upload_folder:
         print("ERROR: UPLOAD_FOLDER not configured, cannot delete image.")
         return False
    image_path = os.path.join(upload_folder, image_filename)
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"DEBUG: Admin deleted image: {image_path}")
            return True
        except Exception as e:
            print(f"ERROR deleting image {image_path}: {e}")
            flash(f"删除图片 '{image_filename}' 失败: {e}", 'error')
            return False
    else:
        print(f"WARN: Admin found image path not existing, cannot delete: {image_path}")
        return False
# === ^^^ HELPER FUNCTIONS ^^^ ===

# --- 权限控制装饰器 (已修改) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not (current_user.role == 'admin' or current_user.id == 1):
            flash('您必须以管理员身份登录才能访问此页面。', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES FOR RULE MANAGEMENT (保持不变) ---
# ... (list_rules, add_rule, edit_rule, delete_rule 保持不变) ...
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
    # Pass action='add' and empty form_data for the add form
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

    # Prepare form_data for pre-filling the edit form
    form_data = {
        'id': rule_to_edit.id,
        'category': rule_to_edit.category,
        'question': rule_to_edit.question,
        'answer': rule_to_edit.answer
    }
    # Pass action='edit' and the rule data
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

# --- ROUTES FOR USER MANAGEMENT (已修改) ---
# ... (list_users, view_user, edit_user, reset_user_password, delete_user 保持不变) ...
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

@admin.route('/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user role."""
    user_to_edit = User.query.get_or_404(user_id)

    if request.method == 'POST':
        new_role = request.form.get('role')
        allowed_roles = ['buyer', 'seller', 'admin']

        if new_role in allowed_roles:
            is_last_admin = False
            if (user_to_edit.role == 'admin' or user_to_edit.id == 1) and new_role != 'admin':
                admin_role_count = User.query.filter_by(role='admin').count()
                effective_admin_count = admin_role_count
                user_1 = User.query.get(1)
                if user_1 and user_1.role != 'admin' and user_to_edit.id != 1:
                    effective_admin_count += 1

                if effective_admin_count <= 1:
                    is_last_admin = True
                    flash('操作失败：不能将系统中唯一的管理员权限移除！', 'error')
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
@admin_required
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
@admin_required
def delete_user(user_id):
    """Delete a user account, with safety checks."""
    if user_id == current_user.id:
         flash('操作无效：不能删除自己的管理员账户！', 'error')
         return redirect(url_for('admin.list_users'))
    if user_id == 1:
        flash('操作无效：不能删除 ID 为 1 的超级管理员账户！', 'error')
        return redirect(url_for('admin.list_users'))
    user_to_delete = User.query.get_or_404(user_id)
    try:
        product_count = Product.query.filter_by(seller_id=user_id).count()
        if product_count > 0:
            flash(f'删除失败：用户 {user_to_delete.username} 仍有 {product_count} 件关联商品。请先处理这些商品。', 'error')
            return redirect(url_for('admin.list_users'))
    except Exception as e:
         flash(f'检查关联商品时出错: {e}', 'error')
         print(f"ERROR: Checking products failed for user {user_id} before delete: {e}")
         return redirect(url_for('admin.list_users'))
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

# --- ROUTES FOR PRODUCT MANAGEMENT (ADMIN - 已修改) ---
# ... (list_all_products, edit_product, toggle_product_status, delete_product 保持修改后的逻辑) ...
@admin.route('/products')
@login_required
@admin_required
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
@admin_required
def edit_product(product_id):
    """Allow admin to edit any product's details, including image."""
    product_to_edit = Product.query.get_or_404(product_id)

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        remove_image = request.form.get('remove_image')
        error = None
        price = None
        stock = None

        # 基本数据验证
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        if price_str and error is None: # 只有前面没错才继续验证
            try:
                price = float(price_str)
                if price < 0: error = '价格不能为负数。'
            except ValueError: error = '价格必须是有效的数字。'
        if stock_str and error is None: # 只有前面没错才继续验证
            try:
                stock = int(stock_str)
                if stock < 0: error = '库存不能为负数。'
            except ValueError: error = '库存必须是有效的整数。'

        # 图片处理
        new_image_filename = None
        delete_current_image = False
        if error is None:
            image_file = request.files.get('product_image')
            if remove_image:
                delete_current_image = True
                if image_file and image_file.filename != '':
                    flash('您已选择删除当前图片，新上传的图片将被忽略。', 'warning')
            elif image_file and image_file.filename != '':
                new_image_filename = save_product_image(image_file, product_to_edit.image_url)

        if error is None:
            # 更新商品信息
            product_to_edit.name = name
            product_to_edit.description = description
            product_to_edit.price = price
            product_to_edit.stock = stock

            # 更新图片URL
            if delete_current_image:
                if product_to_edit.image_url:
                    if delete_product_image(product_to_edit.image_url):
                        product_to_edit.image_url = None
            elif new_image_filename:
                product_to_edit.image_url = new_image_filename

            try:
                db.session.commit()
                flash(f'商品 "{product_to_edit.name}" (ID: {product_id}) 更新成功！', 'success')
                return redirect(url_for('admin.list_all_products'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生数据库错误: {e}', 'error')
                print(f"ERROR: Updating product {product_id} failed: {e}")
                if new_image_filename and not delete_current_image:
                    delete_product_image(new_image_filename)
        else:
            flash(error, 'error') # 显示验证错误

    return render_template('admin/edit_product.html', product=product_to_edit)


@admin.route('/products/toggle_status/<int:product_id>', methods=['POST'])
@login_required
@admin_required
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
@admin_required
def delete_product(product_id):
    """Permanently delete a product and its image."""
    product_to_delete = Product.query.get_or_404(product_id)
    image_filename_to_delete = product_to_delete.image_url
    product_name = product_to_delete.name
    try:
        db.session.delete(product_to_delete)
        db.session.commit()
        flash(f'商品 "{product_name}" (ID: {product_id}) 已被永久删除。', 'success')
        if image_filename_to_delete:
            delete_product_image(image_filename_to_delete)
    except Exception as e:
        db.session.rollback()
        flash(f'删除商品时发生错误: {e}', 'error')
        print(f"ERROR: Deleting product {product_id} failed: {e}")
    return redirect(url_for('admin.list_all_products'))

# --- EXPORT ROUTE (保持不变) ---
@admin.route('/products/export')
@login_required
@admin_required
def export_products():
    # ... (保持不变) ...
    try:
        # 查询所有商品及关联的卖家用户名
        products_data = db.session.query(
            Product.id,
            Product.name,
            Product.description,
            Product.price,
            Product.stock,
            Product.status,
            Product.image_url, # 包含图片文件名
            User.username.label('seller_username') # 获取卖家用户名并重命名
        ).join(User, Product.seller_id == User.id).order_by(Product.id).all()

        output = io.StringIO()
        writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # 写入表头
        writer.writerow(['ID', 'Name', 'Description', 'Price', 'Stock', 'Status', 'Image Filename', 'Seller Username'])

        # 写入商品数据
        for row in products_data:
            writer.writerow([
                row.id,
                row.name,
                row.description or '',
                row.price,
                row.stock,
                row.status,
                row.image_url or '',
                row.seller_username
            ])

        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"products_export_{timestamp}.csv"

        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )

    except Exception as e:
        flash(f"导出商品数据时出错: {e}", "error")
        print(f"ERROR exporting products: {e}")
        return redirect(url_for('admin.list_all_products'))


# === vvv NEW IMPORT ROUTE vvv ===
@admin.route('/products/import', methods=['POST'])
@login_required
@admin_required
def import_products():
    """从上传的 CSV 文件批量导入商品数据"""
    if 'import_file' not in request.files:
        flash('错误：请求中未找到文件部分。', 'error')
        return redirect(url_for('admin.list_all_products'))

    file = request.files['import_file']

    if file.filename == '':
        flash('错误：未选择任何文件。', 'error')
        return redirect(url_for('admin.list_all_products'))

    if file and file.filename.endswith('.csv'): # 简单检查文件扩展名
        try:
            # 读取文件内容为字符串，并处理可能的 BOM 头
            stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
            # 使用 DictReader 读取 CSV
            csv_reader = csv.DictReader(stream)

            required_columns = ['Name', 'Price', 'Stock', 'Seller Username'] # 核心必填列
            optional_columns = ['Description', 'Status', 'Image Filename']
            header = csv_reader.fieldnames

            # 检查必需的列是否存在
            if not header or not all(col in header for col in required_columns):
                flash(f"CSV 文件格式错误：必须包含列: {', '.join(required_columns)}", 'error')
                return redirect(url_for('admin.list_all_products'))

            products_to_add = []
            errors = []
            success_count = 0
            line_number = 1 # DictReader 从数据行开始计数

            # 缓存卖家信息以减少数据库查询
            seller_cache = {}

            for row in csv_reader:
                line_number += 1 # 实际数据行号 (加上表头是 line_number)
                row_errors = []

                # 提取数据 (使用 .get() 并提供默认值)
                name = row.get('Name', '').strip()
                description = row.get('Description', '').strip()
                price_str = row.get('Price', '').strip()
                stock_str = row.get('Stock', '').strip()
                status = row.get('Status', 'active').strip().lower() # 默认为 active
                image_filename = row.get('Image Filename', '').strip()
                seller_username = row.get('Seller Username', '').strip()

                # --- 数据验证 ---
                if not name: row_errors.append("商品名称不能为空")
                if not price_str: row_errors.append("价格不能为空")
                if not stock_str: row_errors.append("库存不能为空")
                if not seller_username: row_errors.append("卖家用户名不能为空")

                # 类型转换和范围验证
                price = None
                stock = None
                try:
                    price = float(price_str)
                    if price < 0: row_errors.append("价格不能为负数")
                except (ValueError, TypeError):
                    if price_str: row_errors.append("价格必须是有效的数字")

                try:
                    stock = int(stock_str)
                    if stock < 0: row_errors.append("库存不能为负数")
                except (ValueError, TypeError):
                     if stock_str: row_errors.append("库存必须是有效的整数")

                # 状态验证
                if status not in ['active', 'inactive']:
                    row_errors.append(f"无效的状态值 '{status}' (应为 'active' 或 'inactive')")
                    status = 'active' # 出错时默认为 active

                # 卖家验证 (使用缓存)
                seller_id = None
                if seller_username:
                    if seller_username in seller_cache:
                        seller_id = seller_cache[seller_username]
                    else:
                        seller = User.query.filter_by(username=seller_username).first()
                        if seller and seller.role == 'seller':
                            seller_id = seller.id
                            seller_cache[seller_username] = seller_id # 缓存结果
                        else:
                             row_errors.append(f"未找到或无效的卖家用户名 '{seller_username}'")
                # else: # 卖家用户名为空已在前面检查

                 # 图片文件名验证 (只检查文件名格式，可选检查文件是否存在)
                image_url_to_save = None
                if image_filename:
                    secured_filename = secure_filename(image_filename)
                    if secured_filename != image_filename:
                        row_errors.append(f"图片文件名 '{image_filename}' 包含无效字符，已修正为 '{secured_filename}' (将使用修正后的名称)")
                        image_filename = secured_filename # 使用安全的文件名

                    # (可选但推荐) 检查图片文件是否真的存在于上传目录
                    upload_folder = current_app.config.get('UPLOAD_FOLDER')
                    if upload_folder:
                        image_path = os.path.join(upload_folder, image_filename)
                        if not os.path.exists(image_path):
                            row_errors.append(f"警告：图片文件 '{image_filename}' 在服务器上传目录中未找到，商品将不关联图片")
                        else:
                             image_url_to_save = image_filename # 文件存在，准备保存文件名
                    else:
                         row_errors.append("错误：服务器未配置图片上传目录，无法验证图片文件")

                # --- 验证结束 ---

                if not row_errors:
                    # 如果没有错误，创建 Product 对象准备添加
                    products_to_add.append(Product(
                        name=name,
                        description=description,
                        price=price,
                        stock=stock,
                        status=status,
                        seller_id=seller_id,
                        image_url=image_url_to_save # 使用验证/修正后的文件名
                    ))
                    success_count += 1
                else:
                    # 记录错误
                    errors.append(f"第 {line_number} 行错误: {'; '.join(row_errors)}")

            # --- 循环结束，开始数据库操作 ---
            if products_to_add:
                try:
                    # 批量添加 (如果数据量很大，分批添加可能更好)
                    db.session.add_all(products_to_add)
                    db.session.commit()
                    flash(f"成功导入 {success_count} 件商品。", "success")
                except Exception as e:
                    db.session.rollback()
                    flash(f"导入过程中发生数据库错误: {e}", "error")
                    print(f"ERROR during bulk insert/commit: {e}")
                    # 记录所有行为失败，因为事务回滚了
                    errors.append(f"数据库错误导致所有 {success_count} 个准备导入的商品失败。")
                    success_count = 0 # 重置成功计数
            elif not errors:
                 flash("文件中没有可导入的有效商品数据。", "warning")


            # 显示错误信息
            if errors:
                error_limit = 10 # 最多显示10条错误
                flash(f"导入过程中遇到 {len(errors)} 个错误:", "danger")
                for i, error_msg in enumerate(errors):
                    if i < error_limit:
                        flash(f"- {error_msg}", "danger")
                    elif i == error_limit:
                         flash(f"- ... (还有 {len(errors) - error_limit} 个错误未显示)", "danger")
                         break

        except FileNotFoundError:
             flash("错误：未找到上传的文件。", "error")
        except UnicodeDecodeError:
             flash("错误：文件编码不是有效的 UTF-8。请确保文件以 UTF-8 (无 BOM) 格式保存。", "error")
        except Exception as e:
            flash(f"处理 CSV 文件时发生未知错误: {e}", "error")
            print(f"ERROR processing CSV file: {e}")
            # 如果在读取或解析时出错，可能需要回滚部分已添加的内容（如果分批提交）
            # db.session.rollback() # 简单起见，如果一次性提交，可以在这里回滚
    else:
        flash("错误：请上传有效的 .csv 文件。", "error")

    return redirect(url_for('admin.list_all_products'))

# === ^^^ NEW IMPORT ROUTE ^^^ ===