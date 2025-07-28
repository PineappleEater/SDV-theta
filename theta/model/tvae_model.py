#!/usr/bin/env python3
"""
TVAE 模型 - 基于变分自编码器的表格数据合成
适用场景: 需要潜在空间表示的数据生成，适合数据降维和特征学习
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import TVAESynthesizer
from utils import *

def main():
    print_model_info(
        "TVAE", 
        "基于变分自编码器的表格数据合成模型，通过潜在空间学习数据表示"
    )
    
    # 数据路径
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/tvae"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 预处理数据 (采样5000行，快速训练)
        df_processed = preprocess_data(df, sample_size=5000, reduce_cardinality=True)
        
        # 3. 创建元数据
        metadata = create_metadata(df_processed)
        
        # 4. 创建 TVAE 合成器 (快速配置)
        print("创建 TVAE 合成器...")
        synthesizer = TVAESynthesizer(
            metadata=metadata,
            epochs=20,                    # 降低到20轮训练
            batch_size=500,               # 批处理大小
            encoder_dim=(64, 64),         # 编码器维度
            decoder_dim=(64, 64),         # 解码器维度
            compress_dims=(64, 64),       # 压缩维度
            decompress_dims=(64, 64),     # 解压缩维度
            l2scale=1e-5,                 # L2正则化
            verbose=True                  # 显示训练进度
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
            "TVAE"
        )
        
        # 7. 保存结果
        save_results(
            synthetic_data, 
            "TVAE", 
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
        
        print(f"\n✅ TVAE 模型执行完成！")
        print(f"结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 