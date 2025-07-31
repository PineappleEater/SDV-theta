#!/usr/bin/env python3
"""
序列专用 PAR 模型 - 基于概率自回归的时间序列生成模型 📈
✨ 序列建模: 专门针对时间序列数据设计，处理序列依赖关系
适用场景: 时间序列数据，具有明确的时间顺序和序列依赖关系
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from utils import *
import pandas as pd
import numpy as np

def main():
    print_model_info(
        "Sequential PAR", 
        "📈 序列专用概率自回归模型：时间序列建模 + 智能序列处理 + 序列依赖优化"
    )
    
    # 📈 序列模型专用配置参数
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/par"
    sample_size = 12000  # 统一的采样大小
    num_sequences = 50   # 生成序列数量（PAR专用）
    sequence_length = 40 # 每个序列长度
    epochs = 50          # PAR训练轮数
    target_user_id = 169 # 目标用户ID
    
    try:
        # 1. 智能序列数据加载和预处理
        print("\n🔄 步骤1: 智能序列数据处理")
        
        # 加载数据
        df = load_data(data_path)
        
        # 序列数据预处理（单用户数据）
        df_processed = preprocess_sequential_data(
            df, 
            user_col='user_id',
            target_user_id=target_user_id,
            sample_size=sample_size
        )
        
        if len(df_processed) == 0:
            raise ValueError("预处理后数据为空")
        
        print(f"📊 ✓ 序列预处理完成，最终数据: {df_processed.shape}")
        
        # 2. 创建序列元数据
        print("\n🔄 步骤2: 创建序列元数据")
        metadata = create_sequential_metadata(df_processed)
        
        # 3. 初始化PAR合成器
        print("\n🔄 步骤3: 初始化PAR合成器")
        synthesizer = PARSynthesizer(
            metadata=metadata,
            epochs=epochs,
            verbose=True
        )
        
        print(f"📊 ✓ PAR合成器初始化完成 (epochs={epochs})")
        
        # 4. 带进度条的模型训练
        print("\n🔄 步骤4: 开始训练...")
        
        start_time = datetime.now()
        with progress_bar(total=epochs, desc="训练PAR模型") as pbar:
            # PAR的fit方法没有callback，我们使用简单的进度指示
            synthesizer.fit(df_processed)
            pbar.update(epochs)  # 训练完成后更新进度条
        
        train_time = datetime.now() - start_time
        print(f"📊 ✓ 训练完成！耗时: {train_time}")
        
        # 5. 带进度条的数据生成
        print(f"\n🔄 步骤5: 生成 {num_sequences} 个序列...")
        
        with progress_bar(total=num_sequences, desc="生成合成数据") as pbar:
            synthetic_data = synthesizer.sample(num_sequences=num_sequences)
            pbar.update(num_sequences)
        
        print(f"📊 ✓ 数据生成完成！生成数据: {synthetic_data.shape}")
        
        # 6. 模型评估
        print("\n🔄 步骤6: 模型质量评估")
        quality_report = evaluate_model(
            real_data=df_processed,
            synthetic_data=synthetic_data,
            metadata=metadata,
            model_name="Enhanced PAR"
        )
        
        # 7. 保存结果和报告
        print("\n🔄 步骤7: 保存结果")
        summary_file = save_results(
            synthetic_data=synthetic_data,
            model_name="PAR",
            output_dir=output_dir,
            metadata=metadata,
            quality_report=quality_report,
            train_time=train_time
        )
        
        # 8. 生成详细报告
        print("\n🔄 步骤8: 生成详细报告")
        detailed_report_file = os.path.join(output_dir, "PAR_detailed_report.md")
        with open(detailed_report_file, 'w', encoding='utf-8') as f:
            f.write("# Enhanced PAR 模型详细报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 基本信息\n")
            f.write(f"- **模型类型**: Enhanced PAR (概率自回归模型)\n")
            f.write(f"- **目标用户**: {target_user_id}\n") 
            f.write(f"- **训练数据**: {df_processed.shape[0]} 行 × {df_processed.shape[1]} 列\n")
            f.write(f"- **生成数据**: {synthetic_data.shape[0]} 行 × {synthetic_data.shape[1]} 列\n")
            f.write(f"- **训练时间**: {train_time}\n")
            f.write(f"- **训练轮数**: {epochs}\n\n")
            
            if quality_report:
                overall_score = quality_report.get_score() * 100
                f.write("## 📈 质量评估\n")
                f.write(f"- **总体质量分数**: {overall_score:.2f}%\n")
                
                if overall_score >= 90:
                    quality_level = "🏆 卓越"
                elif overall_score >= 80:
                    quality_level = "🥇 优秀"
                elif overall_score >= 70:
                    quality_level = "🥈 良好"
                elif overall_score >= 60:
                    quality_level = "🥉 一般"
                else:
                    quality_level = "⚠️ 需要改进"
                
                f.write(f"- **质量等级**: {quality_level}\n\n")
            
            f.write("## 📋 数据统计\n")
            f.write("### 真实数据统计\n")
            f.write("```\n")
            f.write(df_processed.describe().to_string())
            f.write("\n```\n\n")
            
            f.write("### 合成数据统计\n")
            f.write("```\n")
            f.write(synthetic_data.describe().to_string())
            f.write("\n```\n\n")
            
            f.write("## 🔍 数据样例\n")
            f.write("### 真实数据样例\n")
            f.write("```\n")
            f.write(df_processed.head(10).to_string())
            f.write("\n```\n\n")
            
            f.write("### 合成数据样例\n")
            f.write("```\n")
            f.write(synthetic_data.head(10).to_string())
            f.write("\n```\n")
        
        print(f"📊 ✓ 详细报告已保存至: {detailed_report_file}")
        
        # 9. 最终总结
        print("\n" + "="*60)
        print("🎉 Enhanced PAR 模型执行完成!")
        print("="*60)
        print(f"📊 训练数据: {df_processed.shape}")
        print(f"📊 生成数据: {synthetic_data.shape}")
        print(f"⏱️  训练耗时: {train_time}")
        if quality_report:
            print(f"📈 质量分数: {quality_report.get_score()*100:.2f}%")
        print(f"📁 输出目录: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Enhanced PAR 模型执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 