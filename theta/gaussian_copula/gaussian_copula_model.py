#!/usr/bin/env python3
"""
GaussianCopula 模型 - 基于高斯Copula的经典合成数据生成
适用场景: 数据分布较为规整，追求训练速度和稳定性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import GaussianCopulaSynthesizer
from utils import *

def main():
    print_model_info(
        "GaussianCopula", 
        "基于高斯Copula的经典合成数据生成模型，适合结构化数据，训练快速稳定"
    )
    
    # 数据路径
    data_path = "../source_data/th_series_data.csv"
    output_dir = "../output/gaussian_copula"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 预处理数据 (采样10000行用于快速测试)
        df_processed = preprocess_data(df, sample_size=10000)
        
        # 3. 创建元数据
        metadata = create_metadata(df_processed)
        
        # 4. 创建 GaussianCopula 合成器
        print("创建 GaussianCopula 合成器...")
        synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,  # 强制最小最大值约束
            enforce_rounding=True,        # 强制舍入到原始精度
            locales=['zh_CN']            # 支持中文locale
        )
        
        # 5. 训练并生成数据
        synthetic_data, train_time = train_and_sample(
            synthesizer, 
            df_processed, 
            num_rows=1000
        )
        
        # 6. 评估模型
        quality_report = evaluate_model(
            df_processed, 
            synthetic_data, 
            metadata, 
            "GaussianCopula"
        )
        
        # 7. 保存结果
        save_results(
            synthetic_data, 
            "GaussianCopula", 
            output_dir, 
            metadata, 
            quality_report, 
            train_time
        )
        
        # 8. 打印详细统计
        print("\n=== 详细统计对比 ===")
        print("\n原始数据统计:")
        print(df_processed.describe())
        print("\n合成数据统计:")
        print(synthetic_data.describe())
        
        print(f"\n✅ GaussianCopula 模型执行完成！")
        print(f"结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 