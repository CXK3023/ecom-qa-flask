# app/routes/seller.py
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, current_app # 添加 current_app
from flask_login import login_required, current_user
from ..models import Product, User
from .. import db
from functools import wraps
# === vvv NEW IMPORTS vvv ===
import os
from werkzeug.utils import secure_filename
# === ^^^ NEW IMPORTS ^^^ ===


seller = Blueprint('seller', __name__)

# === vvv NEW HELPER FUNCTION vvv ===
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_product_image(file, current_image_filename=None):
    """
    处理上传的图片文件：验证、保存、删除旧文件（如果需要）。
    返回保存后的文件名，如果失败或无有效文件则返回 None。
    如果上传了新文件，会尝试删除 current_image_filename 对应的旧文件。
    """
    if not file or file.filename == '':
        return None # 没有上传文件

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 注意：为了防止文件名冲突，实践中通常会重命名文件（例如加时间戳或UUID）
        # filename = str(uuid.uuid4()) + "_" + filename
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

        # --- 删除旧图片 (如果提供了旧文件名且新文件名不同) ---
        if current_image_filename and current_image_filename != filename:
            old_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], current_image_filename)
            if os.path.exists(old_image_path):
                try:
                    os.remove(old_image_path)
                    print(f"DEBUG: Deleted old image: {old_image_path}")
                except Exception as e:
                    print(f"ERROR deleting old image {old_image_path}: {e}")
                    # 不阻塞主流程，但记录错误
                    flash(f"删除旧图片 '{current_image_filename}' 失败: {e}", 'warning')
            else:
                 print(f"WARN: Old image path not found: {old_image_path}")


        # --- 保存新图片 ---
        try:
            # 确保上传目录存在
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(save_path)
            print(f"DEBUG: Image saved to {save_path}")
            return filename # 返回保存的文件名
        except Exception as e:
            flash(f"图片保存失败: {e}", "error")
            print(f"ERROR saving image to {save_path}: {e}")
            return None
    else:
        # 文件类型不允许
        allowed_formats = ", ".join(current_app.config['ALLOWED_EXTENSIONS'])
        flash(f'上传失败：只允许上传 {allowed_formats} 格式的图片。', 'error')
        return None

def delete_product_image(image_filename):
    """删除指定的商品图片文件"""
    if not image_filename:
        return False
    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image_filename)
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"DEBUG: Deleted image: {image_path}")
            return True
        except Exception as e:
            print(f"ERROR deleting image {image_path}: {e}")
            flash(f"删除图片 '{image_filename}' 失败: {e}", 'error')
            return False
    else:
        print(f"WARN: Image path not found, cannot delete: {image_path}")
        return False

# === ^^^ NEW HELPER FUNCTIONS ^^^ ===


# --- 自定义装饰器：检查用户是否为商家 ---
def seller_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'seller':
            flash('您需要以商家身份登录才能访问此页面。', 'error')
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# --- 商家仪表盘路由 ---
@seller.route('/dashboard')
@login_required
@seller_required
def dashboard():
    products = Product.query.filter_by(seller_id=current_user.id).order_by(Product.id.desc()).all()
    return render_template('seller/dashboard.html', products=products)

# --- 添加商品路由 (修改版) ---
@seller.route('/add', methods=['GET', 'POST'])
@login_required
@seller_required
def add_product():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        error = None

        # --- 基本数据验证 (保持不变) ---
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        price = None
        stock = None
        if price_str:
            try:
                price = float(price_str)
                if price < 0: error = '价格不能为负数。'
            except ValueError: error = '价格必须是有效的数字。'
        if stock_str:
            try:
                stock = int(stock_str)
                if stock < 0: error = '库存不能为负数。'
            except ValueError: error = '库存必须是有效的整数。'
        # --- 基本数据验证结束 ---

        # === vvv 图片处理 vvv ===
        image_file = request.files.get('product_image')
        saved_image_filename = None
        if error is None: # 只有在基本数据验证通过后才尝试处理图片
            saved_image_filename = save_product_image(image_file)
            # 如果保存图片失败 (例如格式不对)，save_product_image 会 flash 错误，这里不需要额外处理 error
        # === ^^^ 图片处理 ^^^ ===


        if error is None:
            # 创建新商品对象，关联当前商家
            new_product = Product(
                name=name,
                description=description,
                price=price,
                stock=stock,
                seller_id=current_user.id,
                # === vvv 添加 image_url vvv ===
                image_url=saved_image_filename # 使用保存后的文件名，可能为 None
                # === ^^^ 添加 image_url ^^^ ===
            )
            try:
                db.session.add(new_product)
                db.session.commit()
                flash('商品添加成功！', 'success')
                return redirect(url_for('seller.dashboard'))
            except Exception as e:
                db.session.rollback()
                flash(f'添加商品时出错: {e}', 'error')
                # 如果保存数据库失败，但图片已经保存，需要考虑是否删除已保存的图片
                if saved_image_filename:
                    delete_product_image(saved_image_filename)
        # 注意：如果验证失败 (error is not None)，或者图片处理失败 (flash 消息已发出)，流程会自然走到下面的 return render_template

    # 如果是 GET 请求或 POST 出错，显示添加表单
    # request.form 会自动传递到模板，用于保留用户输入
    return render_template('seller/add_product.html')


# --- 编辑商品路由 (修改版) ---
@seller.route('/edit/<int:product_id>', methods=['GET', 'POST'])
@login_required
@seller_required
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    if product.seller_id != current_user.id:
        flash('您没有权限编辑这个商品。', 'error')
        abort(403)

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        remove_image = request.form.get('remove_image') # 获取 "删除图片" 复选框的值
        error = None

        # --- 基本数据验证 (同添加) ---
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        price = None
        stock = None
        if price_str:
            try:
                price = float(price_str)
                if price < 0: error = '价格不能为负数。'
            except ValueError: error = '价格必须是有效的数字。'
        if stock_str:
            try:
                stock = int(stock_str)
                if stock < 0: error = '库存不能为负数。'
            except ValueError: error = '库存必须是有效的整数。'
        # --- 基本数据验证结束 ---

        # --- 图片处理逻辑 ---
        new_image_filename = None
        delete_current_image = False

        if error is None: # 只有在基本数据验证通过后才处理图片相关操作
            image_file = request.files.get('product_image')

            if remove_image: # 用户勾选了删除当前图片
                delete_current_image = True
                # 如果勾选了删除，即使上传了新文件，我们也优先执行删除
                if image_file and image_file.filename != '':
                    flash('您已选择删除当前图片，新上传的图片将被忽略。', 'warning')
            elif image_file and image_file.filename != '': # 用户没有勾选删除，并且上传了新文件
                # 尝试保存新图片，同时传入当前图片的文件名以便删除旧图
                new_image_filename = save_product_image(image_file, product.image_url)
                # 如果 new_image_filename 为 None，表示保存失败，错误信息已 flash
            # else: 用户没勾选删除，也没上传新文件 -> 保持当前图片不变
        # --- 图片处理逻辑结束 ---

        if error is None:
            # 更新商品对象的属性
            product.name = name
            product.description = description
            product.price = price
            product.stock = stock

            original_image_url = product.image_url # 记录原始 URL，用于错误回滚

            # 更新图片 URL
            if delete_current_image:
                if product.image_url: # 只有当前确实有图片才执行删除
                    if delete_product_image(product.image_url):
                         product.image_url = None # 文件删除成功后，清空数据库记录
                    else:
                        # 文件删除失败，保持数据库记录不变，错误已 flash
                        pass
                else:
                    # 本来就没有图片，无需操作
                    pass
            elif new_image_filename:
                 # 新图片保存成功 (旧图片已在 save_product_image 中尝试删除)
                 product.image_url = new_image_filename # 更新数据库记录为新文件名
            # else: # 保持不变的情况
            #    pass

            try:
                db.session.commit() # 保存所有更改 (包括 image_url)
                flash('商品信息更新成功！', 'success')
                return redirect(url_for('seller.dashboard'))
            except Exception as e:
                db.session.rollback() # 回滚数据库更改
                flash(f'更新商品时发生数据库错误: {e}', 'error')
                # 数据库回滚后，product.image_url 会回到原始值
                # 但如果新图片已保存，旧图片已删除，文件系统状态可能与数据库不一致
                # 理想情况下需要更复杂的事务管理或补偿逻辑，这里简化处理
                if new_image_filename and not delete_current_image:
                    # 尝试删除刚刚保存的新图片，因为它没有被数据库记录
                    delete_product_image(new_image_filename)
        # 注意：如果验证失败 (error is not None)，或者图片处理失败，流程会自然走到下面的 return render_template

    # GET 请求或 POST 请求出错时，显示编辑表单
    # 传递 product 对象用于预填充
    return render_template('seller/edit_product.html', product=product)

# --- 删除商品路由 (修改版) ---
@seller.route('/delete/<int:product_id>', methods=['POST'])
@login_required
@seller_required
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    if product.seller_id != current_user.id:
        flash('您没有权限删除这个商品。', 'error')
        abort(403)

    image_filename_to_delete = product.image_url # 记录图片文件名
    product_name = product.name # 记录商品名

    try:
        db.session.delete(product) # 先从数据库删除记录
        db.session.commit()
        flash(f'商品 "{product_name}" 删除成功！', 'success')
        # 数据库删除成功后，再尝试删除对应的图片文件
        if image_filename_to_delete:
             delete_product_image(image_filename_to_delete) # 调用删除图片的函数
    except Exception as e:
        db.session.rollback()
        flash(f'删除商品时发生数据库错误: {e}', 'error')

    return redirect(url_for('seller.dashboard'))