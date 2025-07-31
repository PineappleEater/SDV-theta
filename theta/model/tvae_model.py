#!/usr/bin/env python3
"""
优化版 TVAE 模型 - 基于表格变分自编码器的深度生成模型 🎯
✨ 重点优化: 根据测评结果大幅改进参数配置（目标：误差从1.81%→<0.8%）
适用场景: 高维数据，注重潜在空间表示和重构质量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import TVAESynthesizer
from utils import *

def main():
    print_model_info(
        "Optimized TVAE", 
        "🎯 重点优化版表格变分自编码器：深度潜在表示 + 智能数据处理 + 变分推理 (目标误差<0.8%)"
    )
    
    # 🔧 优化后的配置参数（基于测评结果调整）
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/tvae"
    sample_size = 12000  # 统一的采样大小
    num_rows_to_generate = 2000
    epochs = 300         # 🚀 大幅增加训练轮数：150→300（解决学习不充分问题）
    
    try:
        # 1. 智能数据加载和预处理
        print("\n🔄 步骤1: 智能数据处理")
        
        # 加载数据
        df = load_data(data_path)
        
        # 智能预处理（单用户数据，TVAE优化配置）
        df_processed = preprocess_data(
            df, 
            user_id=169,              # 使用用户169的数据
            sample_size=sample_size,
            reduce_cardinality=True,  # TVAE需要减少高基数列
            strategy='frequency_based'  # 统一策略确保指标一致性
        )
        
        print(f"📊 ✓ 单用户预处理完成: {df.shape} → {df_processed.shape}")
        
        # 2. 创建元数据和合成器
        print("\n🤖 步骤2: 模型创建和训练")
        
        # 创建元数据
        metadata = create_metadata(df_processed)
        
        # 🎯 创建优化版TVAE合成器（重点改进配置）
        print("🔧 创建优化版TVAE合成器...")
        synthesizer = TVAESynthesizer(
            metadata=metadata,
            epochs=epochs,                    # 🚀 训练轮数：150→300
            batch_size=1000,                  # 🚀 批量大小：500→1000（提高训练稳定性）
            embedding_dim=256,                # 🚀 嵌入维度：128→256（增强表示能力）
            compress_dims=(256, 128, 64),     # 🚀 编码器：(128,128)→(256,128,64)（加深网络）
            decompress_dims=(64, 128, 256),   # 🚀 解码器：(128,128)→(64,128,256)（对称结构）
            l2scale=5e-6,                     # 🚀 L2正则化：1e-5→5e-6（减少过拟合）
            loss_factor=1,                    # 🚀 损失因子：2→1（更平衡的训练）
            # TVAE特有配置 - 优化版
            enforce_min_max_values=True,      # 强制最小最大值约束
            enforce_rounding=True,            # 强制舍入
            verbose=True                      # 详细输出
        )
        
        # 3. 训练模型和生成数据
        print("\n🔄 步骤3: 模型训练和数据生成")
        print("💡 TVAE优化提示: 增加训练轮数和网络深度，预计训练时间较长...")
        synthetic_data, training_time = train_and_sample(
            synthesizer, df_processed, num_rows_to_generate
        )
        
        # 4. 模型评估
        print("\n📋 步骤4: 模型质量评估")
        quality_report = evaluate_model(
            df_processed, synthetic_data, metadata, "Optimized TVAE"
        )
        
        # 5. 保存结果
        print("\n💾 步骤5: 保存结果")
        save_results(
            synthetic_data, quality_report, training_time, 
            output_dir, "TVAE", df_processed
        )
        
        print(f"\n🎉 优化版TVAE模型训练完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"⏱️ 训练时间: {training_time.total_seconds():.2f} 秒")
        print(f"🎯 目标: 将频率误差从1.81%降至<0.8%")
        
    except Exception as e:
        print(f"\n❌ TVAE模型执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 