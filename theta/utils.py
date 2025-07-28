#!/usr/bin/env python3
"""
通用工具函数 - 用于所有SDV模型的数据处理和评估
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据并进行基本预处理"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    return df

def preprocess_data(df, sample_size=None, reduce_cardinality=False):
    """预处理数据"""
    print("正在预处理数据...")
    
    # 如果指定了采样大小，则进行采样以提高训练速度
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"已采样到 {sample_size} 行数据")
    
    # 处理缺失值较多的列
    missing_threshold = 0.8  # 如果缺失值超过80%，删除该列
    high_missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
    if high_missing_cols:
        print(f"删除缺失值过多的列: {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)
    
    # 处理高基数列（为GAN模型优化）
    if reduce_cardinality:
        print("正在处理高基数列...")
        
        # 删除ID列（通常不需要建模）
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            print("已删除id列")
        
        # 处理高基数的分类列
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() > 100:
                if col in ['indicator', 'value']:
                    # 对于indicator和value列，保留最常见的值
                    if col == 'indicator':
                        top_values = df[col].value_counts().head(20).index
                        df.loc[~df[col].isin(top_values), col] = 'other'
                        print(f"已将{col}列高基数值归并为'other'，保留前20个常见值")
                    elif col == 'value':
                        # 对value列进行数值化处理
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # 填充无法转换的值
                            df[col] = df[col].fillna(df[col].median())
                            print(f"已将{col}列转换为数值类型")
                        except:
                            # 如果无法转换，则保留最常见的值
                            top_values = df[col].value_counts().head(50).index
                            df.loc[~df[col].isin(top_values), col] = 'other'
                            print(f"已将{col}列高基数值归并，保留前50个常见值")
    
    # 处理时间列
    time_columns = ['start_time', 'end_time', 'create_time', 'update_time']
    for col in time_columns:
        if col in df.columns:
            # 将时间转换为时间戳（数值形式）
            try:
                df[col] = pd.to_datetime(df[col]).astype(int) // 10**9  # 转换为Unix时间戳
                print(f"已将 {col} 转换为时间戳")
            except:
                print(f"无法转换 {col} 为时间戳，保持原格式")
    
    # 处理混合类型的列
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    print(f"预处理后数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")
    
    # 显示每列的唯一值数量（帮助调试）
    if reduce_cardinality:
        print("\n各列唯一值数量:")
        for col in df.columns:
            print(f"  {col}: {df[col].nunique()} 个唯一值")
    
    return df

def create_metadata(df):
    """创建元数据"""
    print("正在创建元数据...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    print("元数据创建完成")
    return metadata

def train_and_sample(synthesizer, real_data, num_rows=1000):
    """训练合成器并生成数据"""
    print("正在训练合成器...")
    start_time = datetime.now()
    
    synthesizer.fit(real_data)
    
    train_time = datetime.now() - start_time
    print(f"训练完成，耗时: {train_time}")
    
    print(f"正在生成 {num_rows} 行合成数据...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    return synthetic_data, train_time

def evaluate_model(real_data, synthetic_data, metadata, model_name):
    """评估模型性能"""
    print(f"正在评估 {model_name} 模型...")
    
    try:
        quality_report = evaluate_quality(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        
        overall_score = quality_report.get_score()
        print(f"{model_name} 总体质量分数: {overall_score:.2f}%")
        
        return quality_report
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return None

def save_results(synthetic_data, model_name, output_dir, metadata=None, quality_report=None, train_time=None):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合成数据
    output_file = os.path.join(output_dir, f'{model_name}_synthetic_data.csv')
    synthetic_data.to_csv(output_file, index=False)
    print(f"合成数据已保存到: {output_file}")
    
    # 保存摘要报告
    summary_file = os.path.join(output_dir, f'{model_name}_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} 模型摘要报告 ===\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"合成数据形状: {synthetic_data.shape}\n")
        
        if train_time:
            f.write(f"训练时间: {train_time}\n")
        
        if quality_report:
            try:
                score = quality_report.get_score()
                f.write(f"质量分数: {score:.2f}%\n")
            except:
                f.write("质量分数: 无法获取\n")
        
        f.write(f"\n数据统计:\n")
        f.write(synthetic_data.describe().to_string())
        
    print(f"摘要报告已保存到: {summary_file}")

def print_model_info(model_name, description):
    """打印模型信息"""
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"描述: {description}")
    print("=" * 60) 