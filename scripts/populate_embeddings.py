# scripts/populate_embeddings.py
import sys
import os
import logging
from time import time

# 将项目根目录添加到 Python 路径，以便导入 app 相关模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import create_app, db
from app.models import Product
from app.services.embedding_service import generate_embedding

# --- 配置 ---
BATCH_SIZE = 50  # 每次处理和提交的商品数量
# --- 配置结束 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_populate_embeddings():
    """查询没有向量的商品，并为它们生成和保存向量"""
    app = create_app()
    with app.app_context():
        logger.info("Starting embedding population script...")
        processed_count = 0
        error_count = 0
        start_time = time()

        while True:
            try:
                # 查询下一批没有 embedding 的商品
                products_to_process = Product.query.filter(Product.embedding.is_(None)).limit(BATCH_SIZE).all()
            except Exception as e:
                 logger.error(f"Database query failed: {e}", exc_info=True)
                 break # 数据库查询失败，终止脚本

            if not products_to_process:
                logger.info("No more products found without embeddings.")
                break # 没有需要处理的商品了，退出循环

            logger.info(f"Processing batch of {len(products_to_process)} products...")
            batch_start_time = time()

            for product in products_to_process:
                logger.debug(f"Processing product ID: {product.id}, Name: {product.name}")
                text_to_embed = f"商品名称: {product.name}\n商品描述: {product.description or ''}"
                serialized_embedding = None
                try:
                    # 调用服务生成向量
                    serialized_embedding = generate_embedding(text_to_embed)

                    if serialized_embedding:
                        product.embedding = serialized_embedding
                        logger.debug(f"Embedding generated successfully for product ID: {product.id}")
                    else:
                        # generate_embedding 内部出错会返回 None
                        logger.warning(f"Embedding generation returned None for product ID: {product.id}. Skipping.")
                        error_count += 1

                except Exception as e:
                    # 捕获 generate_embedding 可能抛出的未预料错误
                    logger.error(f"Unexpected error generating embedding for product ID {product.id}: {e}", exc_info=True)
                    error_count += 1
                    # 选择跳过此商品，继续处理下一个

            try:
                 # 提交当前批次的更改
                db.session.commit()
                processed_count += len(products_to_process)
                batch_time = time() - batch_start_time
                logger.info(f"Batch committed. Processed {len(products_to_process)} products in {batch_time:.2f} seconds. Total processed: {processed_count}")
            except Exception as e:
                 logger.error(f"Database commit failed for batch: {e}", exc_info=True)
                 db.session.rollback() # 回滚失败的批次
                 error_count += len(products_to_process) # 将整个批次计为错误
                 # 可以在这里决定是停止脚本还是尝试下一批

        end_time = time()
        logger.info(f"Embedding population finished in {end_time - start_time:.2f} seconds.")
        logger.info(f"Total products processed successfully (embedding generated and saved): {processed_count - error_count}")
        logger.info(f"Total errors/skipped products during embedding generation: {error_count}")

if __name__ == "__main__":
    run_populate_embeddings()