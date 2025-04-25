-- 为 ai_model 表添加 invocation_method 列
ALTER TABLE ai_model
ADD COLUMN invocation_method VARCHAR(20) NOT NULL DEFAULT 'local';

-- 为 product 表添加 embedding 列 (使用 LONGBLOB 适应可能较大的向量)
ALTER TABLE product
ADD COLUMN embedding LONGBLOB NULL;