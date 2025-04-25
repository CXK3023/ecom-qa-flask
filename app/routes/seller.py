# app/routes/seller.py
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, current_app # 添加 current_app
from flask_login import login_required, current_user
from ..models import Product, User
from .. import db
from functools import wraps
import os
from werkzeug.utils import secure_filename
# === vvv 新增：导入 embedding 服务 vvv ===
from ..services.embedding_service import generate_embedding
# === ^^^ 新增 ^^^ ===


seller = Blueprint('seller', __name__)

# === vvv HELPER FUNCTIONS (Image Handling - 保持不变) vvv ===
# ... (allowed_file, save_product_image, delete_product_image 函数代码保持不变) ...
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
                    flash(f"删除旧图片 '{current_image_filename}' 失败: {e}", 'warning')
            else:
                 print(f"WARN: Old image path not found: {old_image_path}")


        # --- 保存新图片 ---
        try:
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
# === ^^^ HELPER FUNCTIONS ^^^ ===


# --- 自定义装饰器：检查用户是否为商家 (保持不变) ---
def seller_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'seller':
            flash('您需要以商家身份登录才能访问此页面。', 'error')
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# --- 商家仪表盘路由 (保持不变) ---
@seller.route('/dashboard')
@login_required
@seller_required
def dashboard():
    products = Product.query.filter_by(seller_id=current_user.id).order_by(Product.id.desc()).all()
    return render_template('seller/dashboard.html', products=products)

# --- 添加商品路由 (修改版：添加 Embedding 生成) ---
@seller.route('/add', methods=['GET', 'POST'])
@login_required
@seller_required
def add_product():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description', '') # 获取描述，默认为空字符串
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

        # === 图片处理 (保持不变) ===
        image_file = request.files.get('product_image')
        saved_image_filename = None
        if error is None:
            saved_image_filename = save_product_image(image_file)
        # === 图片处理结束 ===

        # === vvv Embedding 生成 vvv ===
        serialized_embedding = None
        if error is None: # 只有在基本验证通过后才尝试生成
            text_to_embed = f"商品名称: {name}\n商品描述: {description}" # 组合名称和描述
            try:
                serialized_embedding = generate_embedding(text_to_embed)
                if serialized_embedding is None:
                    # generate_embedding 内部出错会返回 None
                    flash('警告：未能成功生成商品内容的向量表示。搜索相关性可能受影响。', 'warning')
                else:
                     flash('商品向量已生成。', 'info') # 可以添加一个成功的提示
            except Exception as e:
                # 捕获 generate_embedding 可能抛出的未预料错误
                flash(f'警告：生成商品向量时发生意外错误: {e}', 'warning')
                print(f"ERROR generating embedding during product add: {e}", exc_info=True)
        # === ^^^ Embedding 生成 ^^^ ===

        if error is None:
            # 创建新商品对象
            new_product = Product(
                name=name,
                description=description,
                price=price,
                stock=stock,
                seller_id=current_user.id,
                image_url=saved_image_filename,
                embedding=serialized_embedding # <--- 保存向量
            )
            try:
                db.session.add(new_product)
                db.session.commit()
                flash('商品添加成功！', 'success')
                return redirect(url_for('seller.dashboard'))
            except Exception as e:
                db.session.rollback()
                flash(f'添加商品时发生数据库错误: {e}', 'error')
                if saved_image_filename:
                    delete_product_image(saved_image_filename)
        # 注意：如果验证失败或图片处理失败或数据库保存失败，流程会自然走到下面的 return render_template

    # GET 请求或 POST 出错
    return render_template('seller/add_product.html')


# --- 编辑商品路由 (修改版：添加 Embedding 更新) ---
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
        description = request.form.get('description', '') # 获取描述
        price_str = request.form.get('price')
        stock_str = request.form.get('stock')
        remove_image = request.form.get('remove_image')
        error = None

        # --- 基本数据验证 (保持不变) ---
        if not name: error = '商品名称不能为空。'
        if not price_str: error = '价格不能为空。'
        if not stock_str: error = '库存不能为空。'
        price = None
        stock = None
        if price_str:
            try: price = float(price_str);
            except ValueError: error = '价格必须是有效的数字。'
            if price is not None and price < 0: error = '价格不能为负数。'
        if stock_str:
            try: stock = int(stock_str);
            except ValueError: error = '库存必须是有效的整数。'
            if stock is not None and stock < 0: error = '库存不能为负数。'
        # --- 基本数据验证结束 ---

        # --- 图片处理逻辑 (保持不变) ---
        new_image_filename = None
        delete_current_image = False
        if error is None:
            image_file = request.files.get('product_image')
            if remove_image:
                delete_current_image = True
                if image_file and image_file.filename != '': flash('您已选择删除当前图片，新上传的图片将被忽略。', 'warning')
            elif image_file and image_file.filename != '':
                new_image_filename = save_product_image(image_file, product.image_url)
        # --- 图片处理逻辑结束 ---

        # === vvv Embedding 更新 vvv ===
        serialized_embedding = None
        if error is None: # 只有在基本验证通过后才尝试更新
            text_to_embed = f"商品名称: {name}\n商品描述: {description}" # 使用更新后的名称和描述
            try:
                serialized_embedding = generate_embedding(text_to_embed)
                if serialized_embedding is None:
                    flash('警告：未能成功更新商品内容的向量表示。搜索相关性可能受影响。', 'warning')
                else:
                     flash('商品向量已更新。', 'info')
            except Exception as e:
                flash(f'警告：更新商品向量时发生意外错误: {e}', 'warning')
                print(f"ERROR generating embedding during product edit (ID: {product_id}): {e}", exc_info=True)
        # === ^^^ Embedding 更新 ^^^ ===

        if error is None:
            # 更新商品对象的属性
            product.name = name
            product.description = description
            product.price = price
            product.stock = stock

            # 更新图片 URL (逻辑不变)
            if delete_current_image:
                if product.image_url and delete_product_image(product.image_url):
                     product.image_url = None
            elif new_image_filename:
                 product.image_url = new_image_filename

            # === vvv 保存更新后的 Embedding vvv ===
            if serialized_embedding is not None: # 只有成功生成才更新
                product.embedding = serialized_embedding
            elif product.embedding is not None: # 如果这次生成失败，但之前有值，可以选择保留旧值或清空
                 # flash('警告：本次向量生成失败，保留之前的向量数据。', 'warning')
                 pass # 这里选择保留旧值（不修改 product.embedding）
            # === ^^^ 保存更新后的 Embedding ^^^ ===

            try:
                db.session.commit() # 保存所有更改
                flash('商品信息更新成功！', 'success')
                return redirect(url_for('seller.dashboard'))
            except Exception as e:
                db.session.rollback()
                flash(f'更新商品时发生数据库错误: {e}', 'error')
                if new_image_filename and not delete_current_image:
                    delete_product_image(new_image_filename)
        # 注意：如果验证失败或处理失败，流程会自然走到下面的 return render_template

    # GET 请求或 POST 请求出错时，显示编辑表单
    return render_template('seller/edit_product.html', product=product)

# --- 删除商品路由 (保持不变) ---
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
        db.session.delete(product) # 从数据库删除记录
        db.session.commit()
        flash(f'商品 "{product_name}" 删除成功！', 'success')
        # 数据库删除成功后，再尝试删除对应的图片文件
        if image_filename_to_delete:
             delete_product_image(image_filename_to_delete)
    except Exception as e:
        db.session.rollback()
        flash(f'删除商品时发生数据库错误: {e}', 'error')

    return redirect(url_for('seller.dashboard'))