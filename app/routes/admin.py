# app/routes/admin.py (Fixed print statement in delete_product)
# === vvv IMPORTS (保持不变) vvv ===
from flask import Blueprint, render_template, abort, flash, redirect, url_for, request, current_app, Response
from flask_login import login_required, current_user
from ..models import FaqRule, User, Product, AiModel, EmbeddingModel, SystemSetting, ProductImage
from .. import db
from functools import wraps
from ..services.data_service import clear_rules_cache
# from ..services.embedding_service import generate_embedding # 在需要的地方导入
from werkzeug.security import generate_password_hash
import os
from werkzeug.utils import secure_filename
import csv
import io
from datetime import datetime
import sys # Import sys for traceback printing
# === ^^^ IMPORTS ^^^ ===

admin = Blueprint('admin', __name__)

# === vvv HELPER FUNCTIONS (保持不变) vvv ===
# ... (allowed_file, save_product_image, delete_product_image 代码不变) ...
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
# ... (list_rules, add_rule, edit_rule, delete_rule 代码不变) ...
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
# ... (list_users, view_user, edit_user, reset_user_password, delete_user 代码不变) ...
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
    temp_password = "Password123" # 仍然使用这个不安全的密码作为示例
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
# === ROUTES FOR PRODUCT MANAGEMENT (ADMIN) ===
# ===================================================
# ... (list_all_products, edit_product, toggle_product_status, export_products, import_products 不变) ...
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
    """Allow admin to edit any product's details, including image and embedding."""
    from ..services.embedding_service import generate_embedding
    product_to_edit = Product.query.get_or_404(product_id)
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description', '')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        remove_image = request.form.get('remove_image') # 假设模板中已有此 checkbox
        error = None; price = None; stock = None
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        if price_str and error is None:
            try: price = float(price_str);
            except ValueError: error = '价格必须是有效的数字。'
            if price is not None and price < 0: error = '价格不能为负数。'
        if stock_str and error is None:
            try: stock = int(stock_str);
            except ValueError: error = '库存必须是有效的整数。'
            if stock is not None and stock < 0: error = '库存不能为负数。'

        images_to_add = []
        image_files = request.files.getlist('product_images') # 获取多文件列表
        if error is None:
            # 处理新上传的图片
            if image_files:
                for image_file in image_files:
                    if image_file and image_file.filename != '':
                        saved_filename = save_product_image(image_file) # 注意：旧的 save_product_image 不处理旧文件删除
                        if saved_filename:
                            images_to_add.append(ProductImage(image_filename=saved_filename))
                        else:
                            error = "至少有一个图片上传失败。" # 可以在 save_product_image 中设置更详细的 flash 消息
                            break # 如果一个失败，可能需要停止处理

            # 处理删除图片请求 (假设模板中为每个图片提供删除按钮或复选框)
            # image_ids_to_delete = request.form.getlist('delete_image_ids') # 获取要删除的图片ID列表
            # if image_ids_to_delete:
            #    for img_id in image_ids_to_delete:
            #        img_to_delete = ProductImage.query.get(int(img_id))
            #        if img_to_delete and img_to_delete.product_id == product_id: # 安全检查
            #            filename_to_delete = img_to_delete.image_filename
            #            db.session.delete(img_to_delete)
            #            delete_product_image(filename_to_delete) # 删除物理文件
            #        else:
            #             flash(f"尝试删除无效或不属于此商品的图片 ID: {img_id}", "warning")

        serialized_embedding = None
        if error is None:
            text_to_embed = f"商品名称: {name}\n商品描述: {description}"
            try:
                serialized_embedding = generate_embedding(text_to_embed)
                if serialized_embedding is None: flash('警告：未能成功更新商品内容的向量表示。', 'warning')
                else: flash('商品向量已更新。', 'info')
            except Exception as e:
                flash(f'警告：更新商品向量时发生意外错误: {e}', 'warning')
                print(f"ERROR embedding admin edit (ID: {product_id}): {e}", file=sys.stderr) # 打印到 stderr

        if error is None:
            product_to_edit.name = name; product_to_edit.description = description
            product_to_edit.price = price; product_to_edit.stock = stock
            if serialized_embedding is not None: product_to_edit.embedding = serialized_embedding

            # 关联新图片
            if images_to_add:
                for new_image in images_to_add:
                    product_to_edit.images.append(new_image) # 添加到关系中

            try:
                db.session.commit()
                flash(f'商品 "{product_to_edit.name}" (ID: {product_id}) 更新成功！', 'success')
                return redirect(url_for('admin.list_all_products'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生数据库错误: {e}', 'error')
                # 如果出错，需要删除刚刚保存但未成功关联的图片文件
                for img in images_to_add:
                     delete_product_image(img.image_filename)
        else: flash(error, 'error')
    # GET 请求时传递现有图片列表给模板
    existing_images = product_to_edit.images.all() # lazy='dynamic' 需要 .all()
    return render_template('admin/edit_product.html', product=product_to_edit, images=existing_images)


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

# --- vvv 修改 delete_product 函数 (移除 print 的 exc_info 参数) vvv ---
@admin.route('/products/delete/<int:product_id>', methods=['POST'])
@login_required
@admin_required
def delete_product(product_id):
    """Permanently delete a product, its associated images, and the image files."""
    product_to_delete = Product.query.get_or_404(product_id)
    product_name = product_to_delete.name # 在删除对象前获取名称

    try:
        # 1. 查询并记录所有关联的 ProductImage 记录及其文件名
        #    注意：如果 Product.images 设置了 cascade='all, delete-orphan',
        #    SQLAlchemy 会在 commit 时自动处理删除关联的 ProductImage 记录。
        #    但为了显式删除物理文件，我们仍然需要先查询它们。
        #    如果 cascade 未设置或无效，则必须手动删除 ProductImage 记录。
        associated_images = ProductImage.query.filter_by(product_id=product_id).all()
        image_filenames_to_delete = [img.image_filename for img in associated_images if img.image_filename]

        # 2. 删除所有关联的 ProductImage 数据库记录 (如果 cascade 未生效)
        #    如果 cascade='all, delete-orphan' 已在 models.py 中设置并生效，
        #    下面这个循环可以省略，SQLAlchemy 会自动处理。
        #    为了保险起见，或者如果 cascade 不确定是否生效，保留此循环是安全的。
        # for image_record in associated_images:
        #    db.session.delete(image_record)
        #    print(f"DEBUG: Explicitly deleting ProductImage record ID: {image_record.id}")

        # 3. 删除 Product 数据库记录 (依赖 cascade 来删除关联的 ProductImage)
        db.session.delete(product_to_delete)
        print(f"DEBUG: Deleting Product record ID: {product_id}")

        # 4. 提交数据库事务 (此时 cascade 生效，删除 ProductImage 记录)
        db.session.commit()
        flash(f'商品 "{product_name}" (ID: {product_id}) 及其关联图片记录已成功从数据库删除。', 'success')

        # 5. 尝试删除物理图片文件 (在数据库提交成功后进行)
        if image_filenames_to_delete:
            print(f"DEBUG: Attempting to delete image files: {image_filenames_to_delete}")
            all_files_deleted = True
            for filename in image_filenames_to_delete:
                if not delete_product_image(filename): # 复用现有的删除文件函数
                    all_files_deleted = False
            if not all_files_deleted:
                flash(f'警告：商品 "{product_name}" 的部分或全部图片文件可能未能成功删除，请检查服务器日志和上传目录。', 'warning')
        else:
             print(f"DEBUG: No associated image files found to delete for product ID: {product_id}")

    except Exception as e:
        db.session.rollback()
        flash(f'删除商品 "{product_name}" (ID: {product_id}) 时发生错误: {e}', 'error')
        # --- vvv 修复 print 语句 vvv ---
        print(f"ERROR deleting product ID {product_id}: {e}", file=sys.stderr) # 打印到 stderr
        import traceback
        traceback.print_exc(file=sys.stderr) # 打印完整的 traceback 到 stderr
        # --- ^^^ 修复 print 语句 ^^^ ---

    return redirect(url_for('admin.list_all_products'))
# --- ^^^ 修改 delete_product 函数 ^^^ ---

@admin.route('/products/export')
@login_required
@admin_required
def export_products():
    """Export all products to a CSV file."""
    try:
        products_data = db.session.query(
            Product.id, Product.name, Product.description, Product.price, Product.stock,
            Product.status, User.username.label('seller_username')
        ).join(User, Product.seller_id == User.id).order_by(Product.id).all()

        # 获取图片信息
        product_images_dict = {}
        all_images = ProductImage.query.all()
        for img in all_images:
            if img.product_id not in product_images_dict:
                product_images_dict[img.product_id] = []
            product_images_dict[img.product_id].append(img.image_filename)

        output = io.StringIO(); writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Name', 'Description', 'Price', 'Stock', 'Status', 'Seller Username', 'Image Filenames'])
        for row in products_data:
            image_filenames = "; ".join(product_images_dict.get(row.id, [])) # 使用分号分隔
            writer.writerow([row.id, row.name, row.description or '', row.price, row.stock, row.status, row.seller_username, image_filenames])

        output.seek(0); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"products_export_{timestamp}.csv"
        return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={filename}"})
    except Exception as e: flash(f"导出商品数据时出错: {e}", "error"); return redirect(url_for('admin.list_all_products'))


@admin.route('/products/import', methods=['POST'])
@login_required
@admin_required
def import_products():
    """Import products from an uploaded CSV file, including embedding generation and image association."""
    from ..services.embedding_service import generate_embedding
    if 'import_file' not in request.files: flash('错误：请求中未找到文件部分。', 'error'); return redirect(url_for('admin.list_all_products'))
    file = request.files['import_file']
    if file.filename == '': flash('错误：未选择任何文件。', 'error'); return redirect(url_for('admin.list_all_products'))
    if file and file.filename.endswith('.csv'):
        try:
            stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
            csv_reader = csv.DictReader(stream)
            required_columns = ['Name', 'Price', 'Stock', 'Seller Username']
            optional_columns = ['Description', 'Status', 'Image Filenames']
            header = csv_reader.fieldnames
            if not header or not all(col in header for col in required_columns):
                flash(f"CSV 文件格式错误：必须包含列: {', '.join(required_columns)}", 'error')
                return redirect(url_for('admin.list_all_products'))

            products_to_add = []; product_image_associations = []; errors = []; success_count = 0; line_number = 1; seller_cache = {}
            successful_imports_details = []

            for row in csv_reader:
                line_number += 1; row_errors = []
                name = row.get('Name', '').strip(); description = row.get('Description', '').strip(); price_str = row.get('Price', '').strip(); stock_str = row.get('Stock', '').strip()
                status = row.get('Status', 'active').strip().lower();
                image_filenames_str = row.get('Image Filenames', '').strip()
                seller_username = row.get('Seller Username', '').strip()

                if not name: row_errors.append("商品名称不能为空"); price = None; stock = None
                if not price_str: row_errors.append("价格不能为空")
                if not stock_str: row_errors.append("库存不能为空")
                if not seller_username: row_errors.append("卖家用户名不能为空")

                try:
                    price = float(price_str)
                    if price is not None and price < 0: row_errors.append("价格不能为负数")
                except ValueError:
                    row_errors.append("价格必须是有效的数字")

                try:
                    stock = int(stock_str)
                    if stock is not None and stock < 0: row_errors.append("库存不能为负数")
                except ValueError:
                    row_errors.append("库存必须是有效的整数")

                if status not in ['active', 'inactive']: row_errors.append(f"无效的状态值 '{status}'"); status = 'active'
                seller_id = None
                if seller_username:
                    if seller_username in seller_cache: seller_id = seller_cache[seller_username]
                    else:
                        seller = User.query.filter_by(username=seller_username).first()
                        if seller and seller.role == 'seller': seller_id = seller.id; seller_cache[seller_username] = seller_id
                        else: row_errors.append(f"未找到或无效的卖家用户名 '{seller_username}'")

                image_filenames_to_check = []
                if image_filenames_str:
                    image_filenames_to_check = [fn.strip() for fn in image_filenames_str.replace(',', ';').split(';') if fn.strip()]

                valid_image_filenames = []
                upload_folder = current_app.config.get('UPLOAD_FOLDER')
                if not upload_folder and image_filenames_to_check:
                     row_errors.append("错误：服务器未配置图片上传目录，无法验证图片文件。")
                elif image_filenames_to_check:
                    for image_filename in image_filenames_to_check:
                         secured_filename = secure_filename(image_filename)
                         if secured_filename != image_filename: row_errors.append(f"警告：图片文件名 '{image_filename}' 包含无效字符，已修正为 '{secured_filename}'"); image_filename = secured_filename
                         image_path = os.path.join(upload_folder, image_filename)
                         if not os.path.exists(image_path): row_errors.append(f"警告：图片文件 '{image_filename}' 在服务器上传目录中未找到。")
                         else: valid_image_filenames.append(image_filename)

                serialized_embedding = None
                if not row_errors:
                    text_to_embed = f"商品名称: {name}\n商品描述: {description}"
                    try:
                        serialized_embedding = generate_embedding(text_to_embed)
                        if serialized_embedding is None: errors.append(f"第 {line_number} 行警告: 商品 '{name}' 的向量生成失败。")
                    except Exception as e:
                        errors.append(f"第 {line_number} 行警告: 商品 '{name}' 生成向量时出错: {e}")
                        print(f"ERROR embedding import (Line {line_number}, Name: {name}): {e}", file=sys.stderr)

                if not row_errors:
                    new_product = Product(
                        name=name, description=description, price=price, stock=stock,
                        status=status, seller_id=seller_id, embedding=serialized_embedding
                    )
                    products_to_add.append(new_product)
                    if valid_image_filenames:
                         product_image_associations.append({'product_ref': new_product, 'filenames': valid_image_filenames})

                    successful_imports_details.append(name)
                    success_count += 1
                else: errors.append(f"第 {line_number} 行错误: {'; '.join(row_errors)}")

            if products_to_add:
                 try:
                    db.session.add_all(products_to_add)
                    db.session.flush()

                    images_to_add = []
                    for assoc in product_image_associations:
                        product_instance = assoc['product_ref']
                        product_id = product_instance.id
                        for filename in assoc['filenames']:
                            images_to_add.append(ProductImage(product_id=product_id, image_filename=filename))

                    if images_to_add:
                        db.session.add_all(images_to_add)

                    db.session.commit()
                    flash(f"成功导入 {success_count} 件商品{f'并关联了 {len(images_to_add)} 张图片' if images_to_add else ''}：{', '.join(successful_imports_details[:5])}{'...' if success_count > 5 else ''}", "success")
                 except Exception as e:
                    db.session.rollback(); flash(f"导入过程中发生数据库错误: {e}", "error")
                    errors.append(f"数据库错误导致 {success_count} 个准备导入的商品失败。"); success_count = 0
            elif not errors: flash("文件中没有可导入的有效商品数据。", "warning")

            if errors:
                 error_limit = 10; flash(f"导入过程中遇到 {len(errors)} 个错误或警告:", "danger")
                 for i, error_msg in enumerate(errors):
                     level = "danger" if "错误:" in error_msg else "warning"; flash(f"- {error_msg}", level)
                     if i >= error_limit -1 and len(errors) > error_limit: flash(f"- ... (还有 {len(errors) - error_limit} 个错误/警告未显示)", "danger"); break
        except Exception as e:
             flash(f"处理 CSV 文件时发生未知错误: {e}", "error")
             print(f"ERROR processing CSV import: {e}", file=sys.stderr)
             import traceback
             traceback.print_exc(file=sys.stderr)
    else: flash("错误：请上传有效的 .csv 文件。", "error")

    return redirect(url_for('admin.list_all_products'))


# ==================================================
# === ROUTES FOR CHAT MODEL MANAGEMENT (保持不变) ===
# ==================================================
# ... (list_chat_models, add_chat_model, edit_chat_model, delete_chat_model, set_active_chat_model 代码不变) ...
@admin.route('/chat_models')
@login_required
@admin_required
def list_chat_models():
    """显示所有 AI 对话模型列表"""
    try:
        all_models = AiModel.query.order_by(AiModel.id).all()
    except Exception as e:
        flash(f"加载对话模型列表时出错: {e}", 'error')
        all_models = []
    return render_template('admin/chat_models_list.html', models=all_models)

@admin.route('/chat_models/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_chat_model():
    """添加新的 AI 对话模型"""
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        error = None
        if not model_name: error = "模型名称 (API Name) 不能为空。"
        else:
            existing_model = AiModel.query.filter_by(model_name=model_name).first()
            if existing_model: error = f"对话模型名称 '{model_name}' 已存在。"
        if error is None:
            try:
                new_model = AiModel(model_name=model_name, display_name=display_name or None, description=description or None)
                db.session.add(new_model)
                db.session.commit()
                flash(f"对话模型 '{model_name}' 添加成功！", 'success')
                return redirect(url_for('admin.list_chat_models'))
            except Exception as e:
                db.session.rollback(); flash(f"添加对话模型时发生数据库错误: {e}", 'error')
        else: flash(error, 'error')
    return render_template('admin/add_edit_chat_model.html', action='add', form_data=request.form)

@admin.route('/chat_models/edit/<int:model_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_chat_model(model_id):
    """编辑现有的 AI 对话模型"""
    model_to_edit = AiModel.query.get_or_404(model_id)
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        error = None
        if not model_name: error = "模型名称 (API Name) 不能为空。"
        else:
            existing_model = AiModel.query.filter(AiModel.model_name == model_name, AiModel.id != model_id).first()
            if existing_model: error = f"对话模型名称 '{model_name}' 已被其他模型使用。"
        if error is None:
            try:
                model_to_edit.model_name = model_name; model_to_edit.display_name = display_name or None; model_to_edit.description = description or None
                db.session.commit(); flash(f"对话模型 (ID: {model_id}) 更新成功！", 'success')
                return redirect(url_for('admin.list_chat_models'))
            except Exception as e:
                db.session.rollback(); flash(f"更新对话模型时发生数据库错误: {e}", 'error')
        else: flash(error, 'error')
    form_data = {'id': model_to_edit.id, 'model_name': request.form.get('model_name', model_to_edit.model_name), 'display_name': request.form.get('display_name', model_to_edit.display_name or ''), 'description': request.form.get('description', model_to_edit.description or '')}
    return render_template('admin/add_edit_chat_model.html', action='edit', form_data=form_data)

@admin.route('/chat_models/delete/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def delete_chat_model(model_id):
    """删除 AI 对话模型"""
    model_to_delete = AiModel.query.get_or_404(model_id)
    if model_to_delete.is_active: flash(f"无法删除：对话模型 '{model_to_delete.model_name}' 是当前激活的模型。", 'error'); return redirect(url_for('admin.list_chat_models'))
    try:
        model_name_deleted = model_to_delete.model_name; db.session.delete(model_to_delete); db.session.commit()
        flash(f"对话模型 '{model_name_deleted}' (ID: {model_id}) 已成功删除！", 'success')
    except Exception as e:
        db.session.rollback(); flash(f"删除对话模型时发生错误: {e}", 'error')
    return redirect(url_for('admin.list_chat_models'))

@admin.route('/chat_models/set_active/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def set_active_chat_model(model_id):
    """将指定对话模型设为活动状态"""
    model_to_activate = AiModel.query.get_or_404(model_id)
    try:
        AiModel.query.update({AiModel.is_active: False}); model_to_activate.is_active = True; db.session.commit()
        flash(f"已成功激活对话模型：'{model_to_activate.model_name}'！", 'success')
    except Exception as e:
        db.session.rollback(); flash(f"激活对话模型时发生错误: {e}", 'error')
    return redirect(url_for('admin.list_chat_models'))

# ===================================================
# === ROUTES FOR EMBEDDING MODEL MANAGEMENT (保持不变) ===
# ===================================================
# ... (list_embedding_models, add_embedding_model, edit_embedding_model, delete_embedding_model, set_active_embedding_model 代码不变) ...
@admin.route('/embedding_models')
@login_required
@admin_required
def list_embedding_models():
    """显示所有 Embedding 模型列表"""
    try:
        all_models = EmbeddingModel.query.order_by(EmbeddingModel.id).all()
    except Exception as e:
        flash(f"加载 Embedding 模型列表时出错: {e}", 'error')
        all_models = []
    return render_template('admin/embedding_models_list.html', models=all_models)

@admin.route('/embedding_models/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_embedding_model():
    """添加新的 Embedding 模型"""
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        invocation_method = request.form.get('invocation_method', 'local').strip()
        error = None
        if not model_name: error = "模型名称不能为空。"
        else:
            existing_model = EmbeddingModel.query.filter_by(model_name=model_name).first()
            if existing_model: error = f"Embedding 模型名称 '{model_name}' 已存在。"
        if invocation_method not in ['local', 'remote_api']: error = "无效的调用方式选择。"
        if error is None:
            try:
                new_model = EmbeddingModel(model_name=model_name, display_name=display_name or None, description=description or None, invocation_method=invocation_method)
                db.session.add(new_model); db.session.commit()
                flash(f"Embedding 模型 '{model_name}' 添加成功！", 'success')
                return redirect(url_for('admin.list_embedding_models'))
            except Exception as e:
                db.session.rollback(); flash(f"添加 Embedding 模型时发生数据库错误: {e}", 'error')
        else: flash(error, 'error')
    return render_template('admin/add_edit_embedding_model.html', action='add', form_data=request.form)

@admin.route('/embedding_models/edit/<int:model_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_embedding_model(model_id):
    """编辑现有的 Embedding 模型"""
    model_to_edit = EmbeddingModel.query.get_or_404(model_id)
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        display_name = request.form.get('display_name', '').strip()
        description = request.form.get('description', '').strip()
        invocation_method = request.form.get('invocation_method', 'local').strip()
        error = None
        if not model_name: error = "模型名称不能为空。"
        else:
            existing_model = EmbeddingModel.query.filter(EmbeddingModel.model_name == model_name, EmbeddingModel.id != model_id).first()
            if existing_model: error = f"Embedding 模型名称 '{model_name}' 已被其他模型使用。"
        if invocation_method not in ['local', 'remote_api']: error = "无效的调用方式选择。"
        if error is None:
            try:
                model_to_edit.model_name = model_name; model_to_edit.display_name = display_name or None; model_to_edit.description = description or None; model_to_edit.invocation_method = invocation_method
                db.session.commit(); flash(f"Embedding 模型 (ID: {model_id}) 更新成功！", 'success')
                return redirect(url_for('admin.list_embedding_models'))
            except Exception as e:
                db.session.rollback(); flash(f"更新 Embedding 模型时发生数据库错误: {e}", 'error')
        else: flash(error, 'error')
    form_data = {'id': model_to_edit.id, 'model_name': request.form.get('model_name', model_to_edit.model_name), 'display_name': request.form.get('display_name', model_to_edit.display_name or ''), 'description': request.form.get('description', model_to_edit.description or ''), 'invocation_method': request.form.get('invocation_method', model_to_edit.invocation_method)}
    return render_template('admin/add_edit_embedding_model.html', action='edit', form_data=form_data)

@admin.route('/embedding_models/delete/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def delete_embedding_model(model_id):
    """删除 Embedding 模型"""
    model_to_delete = EmbeddingModel.query.get_or_404(model_id)
    if model_to_delete.is_active: flash(f"无法删除：Embedding 模型 '{model_to_delete.model_name}' 是当前激活的模型。", 'error'); return redirect(url_for('admin.list_embedding_models'))
    try:
        model_name_deleted = model_to_delete.model_name; db.session.delete(model_to_delete); db.session.commit()
        flash(f"Embedding 模型 '{model_name_deleted}' (ID: {model_id}) 已成功删除！", 'success')
    except Exception as e:
        db.session.rollback(); flash(f"删除 Embedding 模型时发生错误: {e}", 'error')
    return redirect(url_for('admin.list_embedding_models'))

@admin.route('/embedding_models/set_active/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def set_active_embedding_model(model_id):
    """将指定 Embedding 模型设为活动状态"""
    model_to_activate = EmbeddingModel.query.get_or_404(model_id)
    try:
        EmbeddingModel.query.update({EmbeddingModel.is_active: False}); model_to_activate.is_active = True; db.session.commit()
        flash(f"已成功激活 Embedding 模型：'{model_to_activate.model_name}' ({model_to_activate.invocation_method})！", 'success')
    except Exception as e:
        db.session.rollback(); flash(f"激活 Embedding 模型时发生错误: {e}", 'error')
    return redirect(url_for('admin.list_embedding_models'))

# =========================================
# === ROUTES FOR SYSTEM SETTINGS (保持不变) ===
# =========================================
# ... (manage_settings 代码不变) ...
@admin.route('/settings', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_settings():
    """管理系统设置"""
    settings_dict = {}
    try:
        settings = SystemSetting.query.all()
        for setting in settings:
            settings_dict[setting.key] = setting.value
    except Exception as e:
        flash(f'加载系统设置时出错: {e}', 'error')

    if request.method == 'POST':
        settings_to_update = {'enable_vector_search': request.form.get('enable_vector_search')}
        try:
            for key, raw_value in settings_to_update.items():
                value = 'true' if key in request.form else 'false'
                setting = SystemSetting.query.filter_by(key=key).first()
                if setting: setting.value = value
                else: new_setting = SystemSetting(key=key, value=value); db.session.add(new_setting)
            db.session.commit()
            flash('系统设置已成功更新！', 'success')
            return redirect(url_for('admin.manage_settings'))
        except Exception as e:
            db.session.rollback(); flash(f'更新系统设置时发生错误: {e}', 'error')
    return render_template('admin/manage_settings.html', settings=settings_dict)