#!/usr/bin/env python3
"""
PAR 模型 - 基于概率自回归的序列数据合成
适用场景: 时间序列数据，具有序列依赖关系的数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata
from utils import *
import pandas as pd

def prepare_sequential_data(df):
    """为PAR模型准备序列数据"""
    print("正在为PAR模型准备序列数据...")
    
    # 确保有用户ID列用于分组序列
    if 'user_id' not in df.columns:
        print("❌ 数据中没有user_id列，PAR模型需要序列标识符")
        return None
    
    # 排序数据以确保时间顺序
    time_cols = ['start_time', 'end_time', 'create_time', 'update_time']
    sort_col = None
    
    for col in time_cols:
        if col in df.columns:
            sort_col = col
            break
    
    if sort_col:
        # 先转换为数值时间戳
        df[sort_col] = pd.to_datetime(df[sort_col], errors='coerce')
        df = df.sort_values(['user_id', sort_col])
        # 转换为Unix时间戳
        df[sort_col] = df[sort_col].astype(int) // 10**9
        print(f"✓ 已按 user_id 和 {sort_col} 排序")
    else:
        df = df.sort_values('user_id')
        print("✓ 已按 user_id 排序")
    
    # 检查每个用户的序列长度
    sequence_lengths = df.groupby('user_id').size()
    print(f"✓ 用户数量: {len(sequence_lengths)}")
    print(f"✓ 平均序列长度: {sequence_lengths.mean():.1f}")
    print(f"✓ 序列长度范围: {sequence_lengths.min()} - {sequence_lengths.max()}")
    
    # 过滤掉序列太短的用户（少于2个记录）
    valid_users = sequence_lengths[sequence_lengths >= 2].index
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    if len(df_filtered) < len(df):
        removed_count = len(df) - len(df_filtered)
        print(f"✓ 已移除 {removed_count} 条记录（来自序列长度<2的用户）")
    
    return df_filtered

def create_sequential_metadata(df):
    """创建序列数据的元数据"""
    print("正在创建序列数据元数据...")
    
    metadata = Metadata()
    metadata.detect_from_dataframe(
        data=df,
        table_name='health_sequences'
    )
    
    # 设置序列相关的元数据
    metadata.set_sequence_key('health_sequences', 'user_id')
    
    # 如果有时间列，设置为序列索引
    time_cols = ['start_time', 'end_time', 'create_time', 'update_time']
    for col in time_cols:
        if col in df.columns:
            try:
                metadata.set_sequence_index('health_sequences', col)
                print(f"✓ 已设置 {col} 为序列索引")
                break
            except:
                continue
    
    print("✓ 序列元数据创建完成")
    return metadata

def main():
    print_model_info(
        "PAR", 
        "基于概率自回归的序列数据合成模型，专门用于时间序列和具有序列依赖关系的数据"
    )
    
    # 数据路径
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/par"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 预处理数据 (PAR模型需要序列数据，采样10000行)
        df_processed = preprocess_data(df, sample_size=10000)
        
        # 3. 准备序列数据
        df_sequential = prepare_sequential_data(df_processed)
        if df_sequential is None:
            print("❌ 无法准备序列数据，退出PAR模型训练")
            return
        
        # 4. 创建序列元数据
        metadata = create_sequential_metadata(df_sequential)
        
        # 5. 创建 PAR 合成器 (快速配置)
        print("创建 PAR 合成器...")
        synthesizer = PARSynthesizer(
            metadata=metadata,
            epochs=15,              # 降低到15轮训练
            context_columns=None,   # 上下文列（可选）
            verbose=True           # 显示训练进度
        )
        
        # 6. 训练并生成数据
        print("正在训练PAR模型（这可能需要较长时间）...")
        start_time = datetime.now()
        
        synthesizer.fit(df_sequential)
        
        train_time = datetime.now() - start_time
        print(f"✓ PAR模型训练完成，耗时: {train_time}")
        
        # 7. 生成合成数据
        print("正在生成合成序列数据...")
        # 为PAR模型，我们通过指定序列数量来生成数据
        num_sequences = min(50, len(df_sequential['user_id'].unique()))
        synthetic_data = synthesizer.sample(num_sequences=num_sequences)
        
        print(f"✓ 成功生成 {len(synthetic_data)} 行合成序列数据")
        
        # 8. 评估模型（对于序列数据，评估会有所不同）
        print("正在评估PAR模型...")
        try:
            # 序列数据的评估可能需要特殊处理
            quality_report = evaluate_model(
                df_sequential, 
                synthetic_data, 
                metadata, 
                "PAR"
            )
        except Exception as e:
            print(f"⚠️  序列数据评估遇到问题: {e}")
            quality_report = None
        
        # 9. 保存结果
        save_results(
            synthetic_data, 
            "PAR", 
            output_dir, 
            metadata, 
            quality_report, 
            train_time
        )
        
        # 10. 打印详细统计
        print("\n=== 序列数据统计对比 ===")
        print(f"\n原始数据:")
        print(f"- 总记录数: {len(df_sequential)}")
        print(f"- 用户数量: {df_sequential['user_id'].nunique()}")
        print(f"- 平均每用户记录数: {len(df_sequential) / df_sequential['user_id'].nunique():.1f}")
        
        print(f"\n合成数据:")
        print(f"- 总记录数: {len(synthetic_data)}")
        print(f"- 用户数量: {synthetic_data['user_id'].nunique()}")
        print(f"- 平均每用户记录数: {len(synthetic_data) / synthetic_data['user_id'].nunique():.1f}")
        
        print(f"\n✅ PAR 模型执行完成！")
        print(f"结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 