# This file is intentionally left blank.
# run.py
from app import create_app # 从我们上面写的 app/__init__.py 里导入那个“工厂”函数

# 调用工厂函数创建应用实例
app = create_app()

if __name__ == '__main__':
    # 启动 Flask 开发服务器
    # debug=True 表示开启调试模式，方便开发，改代码能自动重载，出错信息更详细
    # host='0.0.0.0' 让服务可以被局域网其他设备访问（或者从 Docker 外部访问）
    # port=5000 指定服务监听的端口号
    app.run(debug=True, host='0.0.0.0', port=5000)