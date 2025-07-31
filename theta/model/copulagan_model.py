#!/usr/bin/env python3
"""
优化版 CopulaGAN 模型 - 结合Copula理论与GAN的混合生成模型 🔥
✨ 轻微优化: 根据测评结果微调参数配置（目标：误差从0.60%→<0.4%）
适用场景: 复杂依赖关系数据，结合统计建模与深度学习优势
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.single_table import CopulaGANSynthesizer
from utils import *

def main():
    print_model_info(
        "Optimized CopulaGAN", 
        "🔥 优化版混合Copula+GAN：统计建模 + 深度学习 + 智能数据处理 (目标误差<0.4%)"
    )
    
    # 🔥 轻微优化的配置参数（基于优秀表现微调）
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/copulagan"
    sample_size = 12000  # 统一的采样大小
    num_rows_to_generate = 2000
    epochs = 140         # 🔥 轻微增加训练轮数：120→140（精细调优）
    
    try:
        # 1. 智能数据加载和预处理
        print("\n🔄 步骤1: 智能数据处理")
        
        # 加载数据
        df = load_data(data_path)
        
        # 智能预处理（单用户数据，CopulaGAN优化配置）
        df_processed = preprocess_data(
            df, 
            user_id=169,              # 使用用户169的数据
            sample_size=sample_size,
            reduce_cardinality=True,  # CopulaGAN需要减少高基数列
            strategy='frequency_based'  # 统一策略确保指标一致性
        )
        
        print(f"📊 ✓ 单用户预处理完成: {df.shape} → {df_processed.shape}")
        
        # 2. 创建元数据和合成器
        print("\n🤖 步骤2: 模型创建和训练")
        
        # 创建元数据
        metadata = create_metadata(df_processed)
        
        # 🔥 创建轻微优化版CopulaGAN合成器（基于优秀表现微调）
        print("🔧 创建优化版CopulaGAN合成器...")
        synthesizer = CopulaGANSynthesizer(
            metadata=metadata,
            epochs=epochs,                    # 🔥 训练轮数：120→140
            batch_size=600,                   # 🔥 批量大小：500→600（轻微增加）
            generator_dim=(256, 256),         # 保持生成器维度（已表现良好）
            discriminator_dim=(256, 256),     # 保持判别器维度
            generator_lr=2e-4,                # 保持生成器学习率（无需调整）
            discriminator_lr=2e-4,            # 保持判别器学习率
            discriminator_steps=1,            # 保持判别器步数
            log_frequency=True,               # 启用日志频率
            verbose=True,                     # 详细输出
            pac=10                           # 保持PAC大小
        )
        
        # 3. 训练模型和生成数据
        print("\n🔄 步骤3: 模型训练和数据生成")
        print("💡 CopulaGAN优化提示: 基于优秀表现进行精细调优，保持稳定性...")
        synthetic_data, training_time = train_and_sample(
            synthesizer, df_processed, num_rows_to_generate
        )
        
        # 4. 模型评估
        print("\n📋 步骤4: 模型质量评估")
        quality_report = evaluate_model(
            df_processed, synthetic_data, metadata, "Optimized CopulaGAN"
        )
        
        # 5. 保存结果
        print("\n💾 步骤5: 保存结果")
        save_results(
            synthetic_data, quality_report, training_time, 
            output_dir, "CopulaGAN", df_processed
        )
        
        print(f"\n🎉 优化版CopulaGAN模型训练完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"⏱️ 训练时间: {training_time.total_seconds():.2f} 秒")
        print(f"🔥 目标: 将频率误差从0.60%降至<0.4%")
        
    except Exception as e:
        print(f"\n❌ CopulaGAN模型执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 