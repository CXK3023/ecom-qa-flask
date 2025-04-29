# visualize_results.py
# Version 4.4 Adaptation - Keyword Excluded

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RESULTS_DIR = 'simulations' # 结果文件所在的目录
RESULT_FILES = {
    'Vector': 'simplified_vector_simulation_results_v4.4.json',
    'RAG': 'rag_simulation_results_v4.4.json',
    # <<< MODIFIED: Comment out Keyword entry >>>
    # 'Keyword': 'keyword_simulation_results_v4.4.json'
}
TOP_K = 3 # 我们主要关注 Top 3 结果
OUTPUT_DIR = 'visualizations' # 图表保存目录

# --- 辅助函数：计算指标 ---
# (calculate_hit_rate_at_k, calculate_mrr_at_k, calculate_distractor_recall_at_k 函数保持不变)
def calculate_hit_rate_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> int:
    """检查 Top-k 结果中是否至少命中一个 GT ID"""
    if not ground_truth_ids: return 0
    gt_set = set(filter(None, ground_truth_ids))
    if not gt_set: return 0
    retrieved_set = set(filter(None, retrieved_ids[:k]))
    return 1 if bool(retrieved_set.intersection(gt_set)) else 0

def calculate_mrr_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """计算 MRR@k"""
    if not ground_truth_ids: return 0.0
    gt_set = set(filter(None, ground_truth_ids))
    if not gt_set: return 0.0
    for i, ret_id in enumerate(retrieved_ids[:k]):
        if ret_id is not None and ret_id in gt_set:
            return 1.0 / (i + 1)
    return 0.0

def calculate_distractor_recall_at_k(retrieved_ids: list, distractor_ids: list, k: int) -> float:
    """计算 Top-k 结果中干扰项的召回比例 (占 Top-k 的比例)"""
    if not distractor_ids: return 0.0
    distractor_set = set(filter(None, distractor_ids))
    if not distractor_set: return 0.0
    retrieved_set = set(filter(None, retrieved_ids[:k]))
    hits = len(retrieved_set.intersection(distractor_set))
    return float(hits) / k if k > 0 else 0.0


# --- 主逻辑 ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    all_results_data = []
    summary_metrics = {}

    # 1. 加载和处理数据 (现在只会加载 Vector 和 RAG)
    for method, filename in RESULT_FILES.items(): # 只会遍历 RESULT_FILES 中未被注释的条目
        filepath = os.path.join(RESULTS_DIR, filename)
        logging.info(f"Processing file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results_list = data.get('results')
            if not isinstance(results_list, list) or not results_list:
                logging.warning(f"'{filename}' has invalid or empty 'results'. Skipping.")
                continue

            df = pd.DataFrame(results_list)
            df['method'] = method

            # --- Retrieval Time Handling ---
            if 'search_time_ms' in df.columns:
                df.rename(columns={'search_time_ms': 'retrieval_time_ms'}, inplace=True)
            elif 'retrieval_time_ms' not in df.columns:
                avg_ret_time = data.get('simulation_metadata', {}).get('average_search_time_ms',
                               data.get('simulation_metadata', {}).get('average_retrieval_time_ms', 0))
                df['retrieval_time_ms'] = avg_ret_time
                logging.warning(f"Missing retrieval time in results for {method}. Using average from metadata: {avg_ret_time}")
            df['retrieval_time_ms'] = pd.to_numeric(df['retrieval_time_ms'], errors='coerce').fillna(0)

            # --- LLM Time Handling ---
            if 'llm_time_ms' not in df.columns:
                 df['llm_time_ms'] = 0
                 logging.warning(f"Missing 'llm_time_ms' in results for {method}. Assigning 0.")
            df['llm_time_ms'] = pd.to_numeric(df['llm_time_ms'], errors='coerce').fillna(0)

            # --- Accuracy Calculation ---
            # 对于 Vector 和 RAG，列名应该是 'retrieved_ids'
            id_col_name = 'retrieved_ids'
            if id_col_name in df.columns:
                df.rename(columns={id_col_name: 'retrieved_ids_for_eval'}, inplace=True)
                logging.info(f"Using column '{id_col_name}' as retrieval source for {method}.")

                required_cols_for_acc = ['retrieved_ids_for_eval', 'ground_truth_ids', 'distractor_ids']
                missing_cols = [col for col in required_cols_for_acc if col not in df.columns]

                if missing_cols:
                    logging.error(f"Missing required columns for accuracy calculation in {filename}: {missing_cols}. Skipping accuracy.")
                    df['hit_rate'] = np.nan
                    df['mrr'] = np.nan
                    df['distractor_recall'] = np.nan
                else:
                    for col in required_cols_for_acc:
                         df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

                    df['hit_rate'] = df.apply(lambda row: calculate_hit_rate_at_k(row['retrieved_ids_for_eval'], row['ground_truth_ids'], TOP_K), axis=1)
                    df['mrr'] = df.apply(lambda row: calculate_mrr_at_k(row['retrieved_ids_for_eval'], row['ground_truth_ids'], TOP_K), axis=1)
                    df['distractor_recall'] = df.apply(lambda row: calculate_distractor_recall_at_k(row['retrieved_ids_for_eval'], row['distractor_ids'], TOP_K), axis=1)
            else:
                logging.error(f"Missing '{id_col_name}' column in {filename}. Cannot calculate accuracy. Skipping accuracy calculation.")
                df['hit_rate'] = np.nan
                df['mrr'] = np.nan
                df['distractor_recall'] = np.nan

            all_results_data.append(df)

            # 计算汇总指标
            summary_metrics[method] = {
                'avg_retrieval_time': df['retrieval_time_ms'].mean(skipna=True),
                'avg_llm_time': df['llm_time_ms'].mean(skipna=True),
                'avg_hit_rate': df['hit_rate'].mean(skipna=True) if 'hit_rate' in df and df['hit_rate'].notna().any() else np.nan,
                'avg_mrr': df['mrr'].mean(skipna=True) if 'mrr' in df and df['mrr'].notna().any() else np.nan,
                'avg_distractor_recall': df['distractor_recall'].mean(skipna=True) if 'distractor_recall' in df and df['distractor_recall'].notna().any() else np.nan,
            }
            logging.info(f"Finished processing {method}. Found {len(df)} results. Calculated means: HitRate={summary_metrics[method]['avg_hit_rate']:.4f}, MRR={summary_metrics[method]['avg_mrr']:.4f}")

        except FileNotFoundError:
            logging.error(f"Result file not found: {filepath}. Skipping.")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {filepath}. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred processing {filepath}: {e}", exc_info=True)

    if not all_results_data:
        logging.error("No valid result data loaded. Cannot generate visualizations.")
        return

    # 合并所有 DataFrame (现在只有 Vector 和 RAG)
    combined_df = pd.concat(all_results_data, ignore_index=True)
    summary_df = pd.DataFrame(summary_metrics).T

    logging.info("\n--- Summary Metrics (Keyword Excluded) ---")
    print(summary_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_no_keyword.csv')) # 保存不含 Keyword 的汇总
    logging.info(f"Summary metrics (Keyword excluded) saved to {os.path.join(OUTPUT_DIR, 'summary_metrics_no_keyword.csv')}")

    # 3. 数据可视化 (绘图代码基本不变，但只会绘制 Vector 和 RAG 的数据)
    sns.set_theme(style="whitegrid")
    try:
        # 尝试设置中文字体 (如果失败，会 fallback)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif'] # 添加更多备选
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as font_ex:
        logging.warning(f"Could not set Chinese font, plots might have missing characters: {font_ex}")


    # a) 对比平均耗时
    plt.figure(figsize=(10, 6))
    # 筛选需要的时间列，重置索引方便 melt
    time_df = summary_df[['avg_retrieval_time', 'avg_llm_time']].reset_index()
    # 使用 melt 将宽表转为长表，方便绘图
    time_df_melted = time_df.melt(id_vars='index', var_name='Time Type', value_name='Average Time (ms)')
    # 重命名列名
    time_df_melted.rename(columns={'index': 'Method'}, inplace=True)
    sns.barplot(x='Method', y='Average Time (ms)', hue='Time Type', data=time_df_melted, palette='viridis')
    plt.title('Vector vs RAG 平均检索与LLM耗时对比') # 更新标题
    plt.ylabel('平均耗时 (毫秒)')
    plt.xlabel('方案')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'average_times_comparison_no_keyword.png'))
    logging.info(f"Saved average times comparison chart (no keyword) to {os.path.join(OUTPUT_DIR, 'average_times_comparison_no_keyword.png')}")
    plt.close()

    # b) 对比准确度指标 (Hit Rate & MRR)
    # 先检查准确度指标是否存在且有效
    if 'avg_hit_rate' in summary_df.columns and summary_df['avg_hit_rate'].notna().any() and \
       'avg_mrr' in summary_df.columns and summary_df['avg_mrr'].notna().any():
        plt.figure(figsize=(10, 6)) # 调整尺寸，因为只有两个方法
        accuracy_df = summary_df[['avg_hit_rate', 'avg_mrr']].reset_index()
        accuracy_df_melted = accuracy_df.melt(id_vars='index', var_name='Metric', value_name='Average Score')
        accuracy_df_melted.rename(columns={'index': 'Method'}, inplace=True)
        metric_map = {'avg_hit_rate': f'Hit Rate @{TOP_K}', 'avg_mrr': f'MRR @{TOP_K}'}
        accuracy_df_melted['Metric'] = accuracy_df_melted['Metric'].map(metric_map)

        sns.barplot(x='Method', y='Average Score', hue='Metric', data=accuracy_df_melted, palette='magma')
        plt.title(f'Vector vs RAG 平均检索准确度指标对比 (Top-{TOP_K})') # 更新标题
        plt.ylabel('平均得分')
        plt.xlabel('方案')
        plt.ylim(0, 1.05) # 留一点顶部空间
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_metrics_comparison_no_keyword.png'))
        logging.info(f"Saved accuracy metrics comparison chart (no keyword) to {os.path.join(OUTPUT_DIR, 'accuracy_metrics_comparison_no_keyword.png')}")
        plt.close()
    else:
        logging.warning("Skipping accuracy metrics plot due to missing or all-NaN data.")


    # c) 对比干扰项召回率
    if 'avg_distractor_recall' in summary_df.columns and summary_df['avg_distractor_recall'].notna().any():
        plt.figure(figsize=(8, 5))
        distractor_df = summary_df[['avg_distractor_recall']].reset_index()
        distractor_df.rename(columns={'index': 'Method', 'avg_distractor_recall': 'Avg Distractor Ratio'}, inplace=True)
        sns.barplot(x='Method', y='Avg Distractor Ratio', data=distractor_df, palette='coolwarm')
        plt.title(f'Vector vs RAG Top-{TOP_K} 结果中平均干扰项比例') # 更新标题
        plt.ylabel(f'平均比例 (Top-{TOP_K})')
        plt.xlabel('方案')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'distractor_recall_comparison_no_keyword.png'))
        logging.info(f"Saved distractor recall comparison chart (no keyword) to {os.path.join(OUTPUT_DIR, 'distractor_recall_comparison_no_keyword.png')}")
        plt.close()
    else:
         logging.warning("Skipping distractor recall plot due to missing or all-NaN data.")

    # d) 耗时分布 (箱形图)
    plt.figure(figsize=(10, 7)) # 调整尺寸
    sns.boxplot(x='method', y='retrieval_time_ms', data=combined_df, palette='viridis') # combined_df 现在只含 V/R
    plt.title('Vector vs RAG 检索耗时分布') # 更新标题
    plt.ylabel('耗时 (毫秒)')
    plt.xlabel('方案')
    # 根据数据范围决定是否用对数刻度
    if combined_df['retrieval_time_ms'].max() / combined_df['retrieval_time_ms'].min() > 50: # 差异较大时用对数
        plt.yscale('log')
        logging.info("Using log scale for retrieval time distribution.")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'retrieval_time_distribution_no_keyword.png'))
    logging.info(f"Saved retrieval time distribution chart (no keyword) to {os.path.join(OUTPUT_DIR, 'retrieval_time_distribution_no_keyword.png')}")
    plt.close()

    plt.figure(figsize=(10, 7)) # 调整尺寸
    sns.boxplot(x='method', y='llm_time_ms', data=combined_df, palette='viridis')
    plt.title('Vector vs RAG LLM调用耗时分布') # 更新标题
    plt.ylabel('耗时 (毫秒)')
    plt.xlabel('方案')
    if combined_df['llm_time_ms'].max() / combined_df['llm_time_ms'].min() > 50:
        plt.yscale('log')
        logging.info("Using log scale for LLM time distribution.")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'llm_time_distribution_no_keyword.png'))
    logging.info(f"Saved LLM time distribution chart (no keyword) to {os.path.join(OUTPUT_DIR, 'llm_time_distribution_no_keyword.png')}")
    plt.close()

    logging.info("--- Visualization script finished (Keyword Excluded) ---")

# ... (calculate functions remain the same) ...

if __name__ == "__main__":
    main()