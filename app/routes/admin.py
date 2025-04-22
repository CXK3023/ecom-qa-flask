# app/routes/admin.py
from flask import Blueprint, render_template, abort, flash, redirect, url_for, request # 导入 request
from flask_login import login_required, current_user
from ..models import FaqRule # 导入 FaqRule 模型
from .. import db
from functools import wraps # 用于自定义装饰器
from ..services.data_service import clear_rules_cache # 导入清除缓存函数

# 创建一个名为 'admin' 的蓝图
admin = Blueprint('admin', __name__)

# --- 权限控制：自定义 admin_required 装饰器 (修改后) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查用户是否已登录，并且角色是否为 'admin'
        if not current_user.is_authenticated or getattr(current_user, 'role', None) != 'admin':
            # 使用 getattr(current_user, 'role', None) 更安全，以防 role 属性不存在
            flash('您必须以管理员身份登录才能访问此页面。', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# --- 路由：显示规则列表 ---
@admin.route('/rules')
@login_required # 必须登录
@admin_required # 必须是管理员
def list_rules():
    """显示所有规则/FAQ 列表"""
    try:
        # 从数据库查询所有规则，按类别和ID排序
        all_rules = FaqRule.query.order_by(FaqRule.category, FaqRule.id).all()
    except Exception as e:
        flash(f'加载规则列表时出错: {e}', 'error')
        all_rules = []

    # 渲染规则列表模板
    # 此时模板中的 url_for('admin.add_rule') 等应该能找到对应的路由了 (但 add/edit/delete 目标页面还未完全实现)
    return render_template('admin/rules_list.html', rules=all_rules)



# --- 路由：添加新规则 ---
@admin.route('/rules/add', methods=['GET', 'POST']) # 定义路由，允许 GET 和 POST
@login_required
@admin_required
def add_rule():
    """显示添加规则表单 (GET) 或处理表单提交 (POST)"""
    if request.method == 'POST':
        # 处理表单提交
        category = request.form.get('category')
        question = request.form.get('question')
        answer = request.form.get('answer')

        error = None
        if not category:
            error = '类别不能为空。'
        elif not question:
            error = '问题不能为空。'
        elif not answer:
            error = '答案不能为空。'

        if error is None:
            # 数据有效，创建新规则
            try:
                new_rule = FaqRule(category=category, question=question, answer=answer)
                db.session.add(new_rule)
                db.session.commit()
                # !!! 重要：清除缓存 !!!
                clear_rules_cache()
                flash('新规则添加成功！', 'success')
                # 添加成功后跳转回规则列表页面
                return redirect(url_for('admin.list_rules'))
            except Exception as e:
                db.session.rollback()
                flash(f'添加规则时发生错误: {e}', 'error')
        else:
            # 显示验证错误
            flash(error, 'error')

    # 如果是 GET 请求，或者 POST 请求有错误，则显示添加/编辑表单
    # 传递 action='add' 告诉模板这是添加操作
    return render_template('admin/add_edit_rule.html', form_data={}, action="add")

# app/routes/admin.py
# ... (保留之前的 imports, Blueprint, admin_required, list_rules, add_rule) ...

# --- 路由：编辑规则 ---
@admin.route('/rules/edit/<int:rule_id>', methods=['GET', 'POST']) # 路由包含规则 ID
@login_required
@admin_required
def edit_rule(rule_id):
    """显示编辑规则表单 (GET) 或处理更新 (POST)"""
    # 使用 get_or_404 查询要编辑的规则，如果 ID 不存在则自动返回 404 错误
    rule_to_edit = FaqRule.query.get_or_404(rule_id) 

    if request.method == 'POST':
        # 处理表单提交
        category = request.form.get('category')
        question = request.form.get('question')
        answer = request.form.get('answer')

        error = None
        if not category:
            error = '类别不能为空。'
        elif not question:
            error = '问题不能为空。'
        elif not answer:
            error = '答案不能为空。'

        if error is None:
            # 数据有效，更新规则对象的属性
            try:
                rule_to_edit.category = category
                rule_to_edit.question = question
                rule_to_edit.answer = answer

                db.session.commit() # 提交更改 (因为对象是从数据库获取的，直接修改后提交即可)
                # !!! 重要：清除缓存 !!!
                clear_rules_cache()
                flash('规则更新成功！', 'success')
                return redirect(url_for('admin.list_rules')) # 更新成功后返回列表页
            except Exception as e:
                db.session.rollback()
                flash(f'更新规则时发生错误: {e}', 'error')
        else:
            # 显示验证错误
            flash(error, 'error')
            # 注意：如果验证失败，再次渲染模板时， Jinja 模板中的
            # request.form.category or form_data.get('category', '')
            # 会优先使用 request.form 中的值，保留用户错误的输入

    # 如果是 GET 请求，或者 POST 请求处理中有错误
    # 需要将当前规则的数据传递给模板用于预填充
    # 我们将 SQLAlchemy 对象转换为字典传递给模板
    form_data = {
        'id': rule_to_edit.id,
        'category': rule_to_edit.category,
        'question': rule_to_edit.question,
        'answer': rule_to_edit.answer
    }
    # 渲染同一个模板，但传入 action='edit' 和包含数据的 form_data
    return render_template('admin/add_edit_rule.html', form_data=form_data, action="edit") 

# app/routes/admin.py
# ... (保留之前的 imports, Blueprint, admin_required, list_rules, add_rule, edit_rule) ...

# --- 路由：删除规则 ---
@admin.route('/rules/delete/<int:rule_id>', methods=['POST']) # 路由包含 ID，只接受 POST 请求
@login_required
@admin_required
def delete_rule(rule_id):
    """处理删除规则的请求"""
    # 查询要删除的规则，找不到则 404
    rule_to_delete = FaqRule.query.get_or_404(rule_id)

    try:
        # 从数据库会话中删除对象
        db.session.delete(rule_to_delete)
        # 提交更改
        db.session.commit()
        # !!! 重要：清除缓存 !!!
        clear_rules_cache()
        flash(f'规则 ID: {rule_id} 已成功删除！', 'success')
    except Exception as e:
        # 如果删除过程中出错，回滚事务
        db.session.rollback()
        flash(f'删除规则时发生错误: {e}', 'error')

    # 无论成功或失败，都重定向回规则列表页面
    return redirect(url_for('admin.list_rules'))