# app/routes/admin.py
# === vvv IMPORTS vvv ===
from flask import Blueprint, render_template, abort, flash, redirect, url_for, request, current_app, Response # 添加 Response
from flask_login import login_required, current_user
# --- 修改：导入 AiModel ---
from ..models import FaqRule, User, Product, AiModel # <--- 添加 AiModel
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

# === vvv HELPER FUNCTIONS (Image Handling - 保持不变) vvv ===
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS')
    if not allowed_extensions:
        print("WARN: ALLOWED_EXTENSIONS not configured in Flask app.")
        return False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_product_image(file, current_image_filename=None):
    """处理上传的图片文件：验证、保存、删除旧文件。返回保存后的文件名或 None。"""
    if not file or file.filename == '': return None
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
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

# --- 权限控制装饰器 (保持不变) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not (current_user.role == 'admin' or current_user.id == 1):
            flash('您必须以管理员身份登录才能访问此页面。', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# === ROUTES FOR RULE MANAGEMENT (保持不变) ===
# ============================================
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
                clear_rules_cache()
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
                clear_rules_cache()
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
        clear_rules_cache()
        flash(f'规则 ID: {rule_id} 已成功删除！', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除规则时发生错误: {e}', 'error')
    return redirect(url_for('admin.list_rules'))

# ============================================
# === ROUTES FOR USER MANAGEMENT (保持不变) ===
# ============================================
@admin.route('/users')
@login_required
@admin_required
def list_users():
    """Display a list of all registered users."""
    try:
        all_users = User.query.order_by(User.id).all()
    except Exception as e:
        flash(f'加载用户列表时出错: {e}', 'error')
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
        except Exception as e:
            flash(f'加载卖家商品列表时出错: {e}', 'error')
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
                if user_1 and user_1.role != 'admin' and user_to_edit.id != 1: effective_admin_count += 1
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
        else: flash('选择的角色无效！', 'error')
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
    return redirect(url_for('admin.list_users'))

@admin.route('/users/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user account, with safety checks."""
    if user_id == current_user.id: flash('操作无效：不能删除自己的管理员账户！', 'error'); return redirect(url_for('admin.list_users'))
    if user_id == 1: flash('操作无效：不能删除 ID 为 1 的超级管理员账户！', 'error'); return redirect(url_for('admin.list_users'))
    user_to_delete = User.query.get_or_404(user_id)
    try:
        product_count = Product.query.filter_by(seller_id=user_id).count()
        if product_count > 0:
            flash(f'删除失败：用户 {user_to_delete.username} 仍有 {product_count} 件关联商品。请先处理这些商品。', 'error')
            return redirect(url_for('admin.list_users'))
    except Exception as e: flash(f'检查关联商品时出错: {e}', 'error'); return redirect(url_for('admin.list_users'))
    try:
        username = user_to_delete.username
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f'用户 {username} (ID: {user_id}) 已被成功删除。', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除用户时发生错误: {e}', 'error')
    return redirect(url_for('admin.list_users'))

# ===================================================
# === ROUTES FOR PRODUCT MANAGEMENT (ADMIN - 不变) ===
# ===================================================
@admin.route('/products')
@login_required
@admin_required
def list_all_products():
    """Display a list of all products from all sellers."""
    try:
        products_with_sellers = db.session.query(Product, User.username)\
                                 .join(User, Product.seller_id == User.id)\
                                 .order_by(Product.status, Product.id.desc()).all()
    except Exception as e:
        flash(f'加载所有商品列表时出错: {e}', 'error')
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
        error = None; price = None; stock = None
        # Basic validation
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        if price_str and error is None:
            try: price = float(price_str);
            except ValueError: error = '价格必须是有效的数字。'
            if price < 0: error = '价格不能为负数。'
        if stock_str and error is None:
            try: stock = int(stock_str);
            except ValueError: error = '库存必须是有效的整数。'
            if stock < 0: error = '库存不能为负数。'
        # Image handling
        new_image_filename = None; delete_current_image = False
        if error is None:
            image_file = request.files.get('product_image')
            if remove_image:
                delete_current_image = True
                if image_file and image_file.filename != '': flash('您已选择删除当前图片，新上传的图片将被忽略。', 'warning')
            elif image_file and image_file.filename != '':
                new_image_filename = save_product_image(image_file, product_to_edit.image_url)
        # Update DB
        if error is None:
            product_to_edit.name = name; product_to_edit.description = description
            product_to_edit.price = price; product_to_edit.stock = stock
            if delete_current_image:
                if product_to_edit.image_url and delete_product_image(product_to_edit.image_url): product_to_edit.image_url = None
            elif new_image_filename: product_to_edit.image_url = new_image_filename
            try:
                db.session.commit()
                flash(f'商品 "{product_to_edit.name}" (ID: {product_id}) 更新成功！', 'success')
                return redirect(url_for('admin.list_all_products'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生数据库错误: {e}', 'error')
                if new_image_filename and not delete_current_image: delete_product_image(new_image_filename)
        else: flash(error, 'error')
    return render_template('admin/edit_product.html', product=product_to_edit)

@admin.route('/products/toggle_status/<int:product_id>', methods=['POST'])
@login_required
@admin_required
def toggle_product_status(product_id):
    """Toggle product status between active and inactive."""
    product_to_toggle = Product.query.get_or_404(product_id)
    try:
        action_text = "上架" if product_to_toggle.status == 'inactive' else "下架"
        product_to_toggle.status = 'active' if product_to_toggle.status == 'inactive' else 'inactive'
        db.session.commit()
        flash(f'商品 "{product_to_toggle.name}" 已成功{action_text}！', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'切换商品状态时发生错误: {e}', 'error')
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
        if image_filename_to_delete: delete_product_image(image_filename_to_delete)
    except Exception as e:
        db.session.rollback()
        flash(f'删除商品时发生错误: {e}', 'error')
    return redirect(url_for('admin.list_all_products'))

@admin.route('/products/export')
@login_required
@admin_required
def export_products():
    """Export all products to a CSV file."""
    try:
        products_data = db.session.query(
            Product.id, Product.name, Product.description, Product.price, Product.stock,
            Product.status, Product.image_url, User.username.label('seller_username')
        ).join(User, Product.seller_id == User.id).order_by(Product.id).all()
        output = io.StringIO(); writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Name', 'Description', 'Price', 'Stock', 'Status', 'Image Filename', 'Seller Username'])
        for row in products_data: writer.writerow([row.id, row.name, row.description or '', row.price, row.stock, row.status, row.image_url or '', row.seller_username])
        output.seek(0); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"products_export_{timestamp}.csv"
        return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={filename}"})
    except Exception as e: flash(f"导出商品数据时出错: {e}", "error"); return redirect(url_for('admin.list_all_products'))

@admin.route('/products/import', methods=['POST'])
@login_required
@admin_required
def import_products():
    """Import products from an uploaded CSV file."""
    if 'import_file' not in request.files: flash('错误：请求中未找到文件部分。', 'error'); return redirect(url_for('admin.list_all_products'))
    file = request.files['import_file']
    if file.filename == '': flash('错误：未选择任何文件。', 'error'); return redirect(url_for('admin.list_all_products'))
    if file and file.filename.endswith('.csv'):
        try:
            stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
            csv_reader = csv.DictReader(stream)
            required_columns = ['Name', 'Price', 'Stock', 'Seller Username']
            header = csv_reader.fieldnames
            if not header or not all(col in header for col in required_columns): flash(f"CSV 文件格式错误：必须包含列: {', '.join(required_columns)}", 'error'); return redirect(url_for('admin.list_all_products'))
            products_to_add = []; errors = []; success_count = 0; line_number = 1; seller_cache = {}
            for row in csv_reader:
                line_number += 1; row_errors = []
                # Extract and validate data... (logic remains the same)
                name = row.get('Name', '').strip(); description = row.get('Description', '').strip(); price_str = row.get('Price', '').strip(); stock_str = row.get('Stock', '').strip()
                status = row.get('Status', 'active').strip().lower(); image_filename = row.get('Image Filename', '').strip(); seller_username = row.get('Seller Username', '').strip()
                if not name: row_errors.append("商品名称不能为空"); price = None; stock = None
                if not price_str: row_errors.append("价格不能为空")
                if not stock_str: row_errors.append("库存不能为空")
                if not seller_username: row_errors.append("卖家用户名不能为空")
                try: price = float(price_str);
                except: row_errors.append("价格必须是有效的数字")
                if price is not None and price < 0: row_errors.append("价格不能为负数")
                try: stock = int(stock_str)
                except: row_errors.append("库存必须是有效的整数")
                if stock is not None and stock < 0: row_errors.append("库存不能为负数")
                if status not in ['active', 'inactive']: row_errors.append(f"无效的状态值 '{status}'"); status = 'active'
                seller_id = None
                if seller_username:
                    if seller_username in seller_cache: seller_id = seller_cache[seller_username]
                    else:
                        seller = User.query.filter_by(username=seller_username).first()
                        if seller and seller.role == 'seller': seller_id = seller.id; seller_cache[seller_username] = seller_id
                        else: row_errors.append(f"未找到或无效的卖家用户名 '{seller_username}'")
                image_url_to_save = None
                if image_filename:
                    secured_filename = secure_filename(image_filename)
                    if secured_filename != image_filename: row_errors.append(f"图片文件名 '{image_filename}' 包含无效字符，已修正为 '{secured_filename}'"); image_filename = secured_filename
                    upload_folder = current_app.config.get('UPLOAD_FOLDER')
                    if upload_folder:
                        image_path = os.path.join(upload_folder, image_filename)
                        if not os.path.exists(image_path): row_errors.append(f"警告：图片文件 '{image_filename}' 在服务器上传目录中未找到")
                        else: image_url_to_save = image_filename
                    else: row_errors.append("错误：服务器未配置图片上传目录")
                # Add product if no errors
                if not row_errors:
                    products_to_add.append(Product(name=name, description=description, price=price, stock=stock, status=status, seller_id=seller_id, image_url=image_url_to_save)); success_count += 1
                else: errors.append(f"第 {line_number} 行错误: {'; '.join(row_errors)}")
            # Commit to DB
            if products_to_add:
                try: db.session.add_all(products_to_add); db.session.commit(); flash(f"成功导入 {success_count} 件商品。", "success")
                except Exception as e: db.session.rollback(); flash(f"导入过程中发生数据库错误: {e}", "error"); errors.append(f"数据库错误导致所有 {success_count} 个准备导入的商品失败。"); success_count = 0
            elif not errors: flash("文件中没有可导入的有效商品数据。", "warning")
            # Flash errors
            if errors:
                error_limit = 10; flash(f"导入过程中遇到 {len(errors)} 个错误:", "danger")
                for i, error_msg in enumerate(errors):
                    if i < error_limit: flash(f"- {error_msg}", "danger")
                    elif i == error_limit: flash(f"- ... (还有 {len(errors) - error_limit} 个错误未显示)", "danger"); break
        except Exception as e: flash(f"处理 CSV 文件时发生未知错误: {e}", "error")
    else: flash("错误：请上传有效的 .csv 文件。", "error")
    return redirect(url_for('admin.list_all_products'))

# ======================================
# === vvv 新增：AI 模型管理路由 vvv ===
# ======================================

@admin.route('/models')
@login_required
@admin_required
def list_models():
    """显示所有 AI 模型列表"""
    try:
        all_models = AiModel.query.order_by(AiModel.id).all()
    except Exception as e:
        flash(f"加载 AI 模型列表时出错: {e}", 'error')
        all_models = []
    return render_template('admin/models_list.html', models=all_models)

@admin.route('/models/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_model():
    """添加新的 AI 模型"""
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        error = None

        if not model_name:
            error = "模型名称 (API Name) 不能为空。"
        else:
            existing_model = AiModel.query.filter_by(model_name=model_name).first()
            if existing_model:
                error = f"模型名称 '{model_name}' 已存在，请使用其他名称。"

        if error is None:
            try:
                new_model = AiModel(
                    model_name=model_name,
                    display_name=display_name if display_name else None,
                    description=description if description else None
                )
                db.session.add(new_model)
                db.session.commit()
                flash(f"AI 模型 '{model_name}' 添加成功！", 'success')
                return redirect(url_for('admin.list_models'))
            except Exception as e:
                db.session.rollback()
                flash(f"添加模型时发生数据库错误: {e}", 'error')
        else:
            flash(error, 'error')

    return render_template('admin/add_edit_model.html', action='add', form_data=request.form)

@admin.route('/models/edit/<int:model_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_model(model_id):
    """编辑现有的 AI 模型"""
    model_to_edit = AiModel.query.get_or_404(model_id)

    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        error = None

        if not model_name:
            error = "模型名称 (API Name) 不能为空。"
        else:
            existing_model = AiModel.query.filter(AiModel.model_name == model_name, AiModel.id != model_id).first()
            if existing_model:
                error = f"模型名称 '{model_name}' 已被其他模型使用。"

        if error is None:
            try:
                model_to_edit.model_name = model_name
                model_to_edit.display_name = display_name if display_name else None
                model_to_edit.description = description if description else None
                db.session.commit()
                flash(f"AI 模型 (ID: {model_id}) 更新成功！", 'success')
                return redirect(url_for('admin.list_models'))
            except Exception as e:
                db.session.rollback()
                flash(f"更新模型时发生数据库错误: {e}", 'error')
        else:
            flash(error, 'error')

    form_data = {
        'id': model_to_edit.id,
        'model_name': request.form.get('model_name', model_to_edit.model_name),
        'display_name': request.form.get('display_name', model_to_edit.display_name or ''),
        'description': request.form.get('description', model_to_edit.description or '')
    }
    return render_template('admin/add_edit_model.html', action='edit', form_data=form_data)

@admin.route('/models/delete/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def delete_model(model_id):
    """删除 AI 模型"""
    model_to_delete = AiModel.query.get_or_404(model_id)

    if model_to_delete.is_active:
        flash(f"无法删除：模型 '{model_to_delete.model_name}' 是当前激活的模型。请先激活其他模型。", 'error')
        return redirect(url_for('admin.list_models'))

    try:
        model_name_deleted = model_to_delete.model_name
        db.session.delete(model_to_delete)
        db.session.commit()
        flash(f"AI 模型 '{model_name_deleted}' (ID: {model_id}) 已成功删除！", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"删除模型时发生错误: {e}", 'error')

    return redirect(url_for('admin.list_models'))

@admin.route('/models/set_active/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def set_active_model(model_id):
    """将指定模型设为活动状态，并将其他模型设为非活动状态"""
    model_to_activate = AiModel.query.get_or_404(model_id)

    try:
        # 使用 SQLAlchemy Core API 或分开查询更新可能更安全，但对于少量模型，这样也可以
        AiModel.query.update({AiModel.is_active: False})
        model_to_activate.is_active = True
        db.session.commit() # 提交两次更新
        flash(f"已成功激活 AI 模型：'{model_to_activate.model_name}'！", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"激活模型时发生错误: {e}", 'error')

    return redirect(url_for('admin.list_models'))

# === ^^^ 新增：AI 模型管理路由 ^^^ ===