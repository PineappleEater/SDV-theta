#!/usr/bin/env python3
"""
优化版 CTGAN 模型 - 基于条件生成对抗网络的深度生成模型 ⚡
✨ 适度优化: 根据测评结果改进参数配置（目标：误差从0.74%→<0.5%）
适用场景: 复杂表格数据，注重条件生成和对抗训练平衡
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import CTGANSynthesizer
from utils import *

def main():
    print_model_info(
        "Optimized CTGAN", 
        "⚡ 优化版条件生成对抗网络：智能数据处理 + 深度对抗训练 + 条件生成 (目标误差<0.5%)"
    )
    
    # ⚡ 优化后的配置参数（适度改进）
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/ctgan"
    sample_size = 12000  # 统一的采样大小
    num_rows_to_generate = 2000
    epochs = 150         # ⚡ 适度增加训练轮数：100→150（提升学习效果）
    
    try:
        # 1. 智能数据加载和预处理
        print("\n🔄 步骤1: 智能数据处理")
        
        # 加载数据
        df = load_data(data_path)
        
        # 智能预处理（单用户数据，CTGAN优化配置）
        df_processed = preprocess_data(
            df, 
            user_id=169,              # 使用用户169的数据
            sample_size=sample_size,
            reduce_cardinality=True,  # CTGAN需要减少高基数列
            strategy='frequency_based'  # 统一策略确保指标一致性
        )
        
        print(f"📊 ✓ 单用户预处理完成: {df.shape} → {df_processed.shape}")
        
        # 2. 创建元数据和合成器
        print("\n🤖 步骤2: 模型创建和训练")
        
        # 创建元数据
        metadata = create_metadata(df_processed)
        
        # ⚡ 创建优化版CTGAN合成器（适度改进配置）
        print("🔧 创建优化版CTGAN合成器...")
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,                    # ⚡ 训练轮数：100→150
            batch_size=750,                   # ⚡ 批量大小：500→750（提高训练效率）
            generator_dim=(256, 256, 128),    # ⚡ 生成器：(256,256)→(256,256,128)（加深网络）
            discriminator_dim=(256, 256),     # 保持判别器维度
            generator_lr=1.5e-4,              # ⚡ 生成器学习率：2e-4→1.5e-4（更稳定训练）
            discriminator_lr=1.5e-4,          # ⚡ 判别器学习率：2e-4→1.5e-4（匹配生成器）
            discriminator_steps=1,            # 保持判别器步数
            log_frequency=True,               # 启用日志频率
            verbose=True,                     # 详细输出
            pac=10                           # 保持PAC大小
        )
        
        # 3. 训练模型和生成数据
        print("\n🔄 步骤3: 模型训练和数据生成")
        print("💡 CTGAN优化提示: 适度增加训练轮数和生成器深度，平衡训练时间...")
        synthetic_data, training_time = train_and_sample(
            synthesizer, df_processed, num_rows_to_generate
        )
        
        # 4. 模型评估
        print("\n📋 步骤4: 模型质量评估")
        quality_report = evaluate_model(
            df_processed, synthetic_data, metadata, "Optimized CTGAN"
        )
        
        # 5. 保存结果
        print("\n💾 步骤5: 保存结果")
        save_results(
            synthetic_data, quality_report, training_time, 
            output_dir, "CTGAN", df_processed
        )
        
        print(f"\n🎉 优化版CTGAN模型训练完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"⏱️ 训练时间: {training_time.total_seconds():.2f} 秒")
        print(f"⚡ 目标: 将频率误差从0.74%降至<0.5%")
        
    except Exception as e:
        print(f"\n❌ CTGAN模型执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 