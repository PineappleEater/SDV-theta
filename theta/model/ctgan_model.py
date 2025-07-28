#!/usr/bin/env python3
"""
CTGAN 模型 - 基于生成对抗网络的深度学习合成数据生成
适用场景: 复杂数据分布，需要高质量合成数据，可接受较长训练时间
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import CTGANSynthesizer
from utils import *

def main():
    print_model_info(
        "CTGAN", 
        "基于生成对抗网络的深度学习模型，能够处理复杂数据分布，生成高质量合成数据"
    )
    
    # 数据路径
    data_path = "../source_data/th_series_data.csv"
    output_dir = "../output/ctgan"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 预处理数据 (CTGAN需要较多数据，采样20000行，处理高基数列)
        df_processed = preprocess_data(df, sample_size=20000, reduce_cardinality=True)
        
        # 3. 创建元数据
        metadata = create_metadata(df_processed)
        
        # 4. 创建 CTGAN 合成器
        print("创建 CTGAN 合成器...")
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=100,                    # 训练轮数
            batch_size=500,               # 批处理大小
            generator_dim=(256, 256),     # 生成器维度
            discriminator_dim=(256, 256), # 判别器维度
            generator_lr=2e-4,            # 生成器学习率
            discriminator_lr=2e-4,        # 判别器学习率
            discriminator_steps=1,        # 每轮训练判别器的步数
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
            "CTGAN"
        )
        
        # 7. 保存结果
        save_results(
            synthetic_data, 
            "CTGAN", 
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
        
        print(f"\n✅ CTGAN 模型执行完成！")
        print(f"结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 