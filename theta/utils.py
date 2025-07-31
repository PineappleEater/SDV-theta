#!/usr/bin/env python3
"""
通用工具函数 - 为所有SDV模型提供统一的数据处理和评估功能
合并了enhanced_utils.py的智能数据处理功能
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
import warnings
warnings.filterwarnings('ignore')

# 进度条支持
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm未安装，将使用简单进度显示")

def progress_bar(iterable=None, total=None, desc="Processing", **kwargs):
    """统一的进度条接口"""
    if TQDM_AVAILABLE:
        if iterable is not None:
            return tqdm(iterable, desc=desc, **kwargs)
        else:
            return tqdm(total=total, desc=desc, **kwargs)
    else:
        # 简单的进度显示类
        class SimpleProgress:
            def __init__(self, total=None, desc="Processing"):
                self.total = total
                self.desc = desc
                self.n = 0
                self.start_time = time.time()
                
            def update(self, n=1):
                self.n += n
                if self.total:
                    percent = (self.n / self.total) * 100
                    elapsed = time.time() - self.start_time
                    print(f"\r{self.desc}: {percent:.1f}% ({self.n}/{self.total}) | 耗时: {elapsed:.1f}s", end="", flush=True)
                else:
                    elapsed = time.time() - self.start_time
                    print(f"\r{self.desc}: {self.n} 项完成 | 耗时: {elapsed:.1f}s", end="", flush=True)
                    
            def close(self):
                print()  # 换行
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                self.close()
        
        if iterable is not None:
            # 对于可迭代对象，创建一个包装器
            class IterableWrapper:
                def __init__(self, iterable, desc):
                    self.iterable = iterable
                    self.desc = desc
                    
                def __iter__(self):
                    with SimpleProgress(total=len(self.iterable) if hasattr(self.iterable, '__len__') else None, desc=self.desc) as pbar:
                        for item in self.iterable:
                            yield item
                            pbar.update(1)
            return IterableWrapper(iterable, desc)
        else:
            return SimpleProgress(total=total, desc=desc)

def load_data(file_path, **kwargs):
    """智能数据加载，支持多种格式和编码"""
    print(f"📊 加载数据: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            # 尝试多种编码和分隔符
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            separators = [',', ';', '\t']
            
            for encoding in progress_bar(encodings, desc="尝试编码"):
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False, **kwargs)
                        if df.shape[1] > 1:  # 确保正确解析了列
                            print(f"📊 ✓ 成功加载 (编码: {encoding}, 分隔符: '{sep}')")
                            break
                    except:
                        continue
                else:
                    continue
                break
            else:
                raise ValueError("无法使用任何编码和分隔符组合加载CSV文件")
                
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        elif file_ext == '.json':
            df = pd.read_json(file_path, **kwargs)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
    except Exception as e:
        raise ValueError(f"数据加载失败: {e}")
    
    print(f"📊 ✓ 数据加载完成: {df.shape}")
    print(f"📊 ✓ 列名: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    
    # 基本数据质量检查
    _basic_data_quality_check(df)
    
    return df

def _basic_data_quality_check(df):
    """基本数据质量检查"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_rate = missing_cells / total_cells * 100
    
    print(f"📊 ✓ 数据质量检查: 缺失率 {missing_rate:.2f}%")
    
    if missing_rate > 50:
        print("📊 ⚠️ 警告: 数据缺失率超过50%，可能影响模型性能")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"📊 ⚠️ 发现 {duplicates} 行重复数据")

def print_model_info(model_name, description):
    """打印模型信息"""
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"描述: {description}")
    print("=" * 60)

def select_single_user(df, user_id=169):
    """选择单个用户的数据"""
    if 'user_id' not in df.columns:
        print("📊 ⚠️ 未找到user_id列，保留所有数据")
        return df
    
    user_data = df[df['user_id'] == user_id].copy()
    if len(user_data) == 0:
        print(f"📊 ⚠️ 未找到用户{user_id}的数据，使用第一个用户的数据")
        first_user = df['user_id'].iloc[0]
        user_data = df[df['user_id'] == first_user].copy()
        user_id = first_user
    
    print(f"📊 ✓ 选择用户{user_id}的数据: {len(user_data)}条记录")
    return user_data

def preprocess_data(df, user_id=169, sample_size=None, reduce_cardinality=False, strategy='frequency_based'):
    """
    智能数据预处理 - 针对单用户数据优化
    
    Args:
        df: 原始DataFrame
        user_id: 用户ID，用于单用户数据处理
        sample_size: 采样大小，如果None则不采样
        reduce_cardinality: 是否减少高基数列（适用于GAN模型）
        strategy: indicator处理策略 ('frequency_based', 'adaptive', 'simple')
    
    Returns:
        预处理后的DataFrame
    """
    print(f"📊 开始单用户智能预处理 (用户ID: {user_id}, 策略: {strategy})")
    original_shape = df.shape
    
    # 1. 选择单用户数据
    df_processed = select_single_user(df, user_id)
    
    # 2. 数据采样（如果需要）
    if sample_size and sample_size < len(df_processed):
        df_processed = df_processed.sample(n=sample_size, random_state=42)
        print(f"📊 ✓ 数据采样: {len(df_processed)} 行")
    
    # 3. 处理缺失值
    df_processed = _handle_missing_values(df_processed)
    
    # 4. 保护数值字段（特别是value字段）
    df_processed = _protect_numeric_fields(df_processed)
    
    # 5. 处理数据类型
    df_processed = _optimize_data_types(df_processed)
    
    # 6. 处理高基数列（如果需要，但保护重要的数值字段）
    if reduce_cardinality:
        df_processed = _reduce_cardinality_smart(df_processed, strategy)
    
    # 7. 处理时间列
    df_processed = _handle_time_columns(df_processed)
    
    # 8. 处理异常值（只对非保护字段）
    df_processed = _handle_outliers_smart(df_processed)
    
    # 9. 最终清理
    df_processed = _final_cleanup(df_processed)
    
    print(f"📊 ✓ 单用户预处理完成: {original_shape} → {df_processed.shape}")
    
    return df_processed

def _handle_missing_values(df):
    """处理缺失值"""
    print("📊 处理缺失值...")
    
    # 删除缺失率超过80%的列
    missing_threshold = 0.8
    high_missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
    if high_missing_cols:
        print(f"📊 ✓ 删除高缺失率列: {len(high_missing_cols)}个")
        df = df.drop(columns=high_missing_cols)
    
    # 填充数值列的缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in progress_bar(numeric_cols, desc="填充数值列"):
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # 填充分类列的缺失值
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in progress_bar(categorical_cols, desc="填充分类列"):
        if df[col].isnull().any():
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('unknown')
    
    return df

def _protect_numeric_fields(df):
    """智能保护和处理数值字段，根据indicator类型优化value字段"""
    print("📊 智能保护和处理数值字段...")
    
    # 定义需要保护的数值字段
    protected_fields = ['value', 'user_id', 'id']
    
    for field in protected_fields:
        if field in df.columns:
            # 确保这些字段保持为数值类型
            if field == 'value':
                # value字段智能处理：根据indicator类型进行不同的处理
                df = _smart_process_value_field(df)
                print(f"📊 ✓ 智能处理数值字段 {field}: {df[field].dtype}")
            elif field in ['user_id', 'id']:
                # ID字段确保为整数
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype('int64')
                print(f"📊 ✓ 保护ID字段 {field}: {df[field].dtype}")
    
    return df

def _smart_process_value_field(df):
    """根据indicator类型智能处理value字段"""
    if 'indicator' not in df.columns or 'value' not in df.columns:
        return df
    
    print("📊 根据indicator类型智能处理value字段...")
    
    # 先尝试转换为数值
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # 处理JSON格式的value
    mask = df['value'].isna()
    if mask.any():
        for idx in df.index[mask]:
            original_value = str(df.at[idx, 'value'])
            if 'primary_value' in original_value:
                try:
                    import json
                    value_dict = json.loads(original_value.replace("'", '"'))
                    if 'primary_value' in value_dict:
                        df.at[idx, 'value'] = float(value_dict['primary_value'])
                except:
                    df.at[idx, 'value'] = 0.0
            else:
                df.at[idx, 'value'] = 0.0
    
    # 根据indicator类型进行特殊处理
    for idx in df.index:
        indicator = str(df.at[idx, 'indicator']).lower()
        value = df.at[idx, 'value']
        
        if pd.isna(value):
            continue
            
        # 1. 时间相关指标统一处理
        if any(keyword in indicator for keyword in ['_start_time', '_end_time', 'sleep_start', 'sleep_end', '_time_', 'time_avg', 'avg_start', 'avg_end']):
            if value > 1000000000:  # Unix时间戳
                try:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(value)
                    df.at[idx, 'value'] = dt.hour  # 转换为小时 (0-23)
                    print(f"📊   时间戳转换: {indicator} {value} → 小时{dt.hour}")
                except:
                    df.at[idx, 'value'] = 12  # 默认中午12点
            elif value > 86400:  # 超过24小时的秒数，转换为小时
                df.at[idx, 'value'] = (value % 86400) / 3600  # 转换为小时
            elif value > 24:  # 可能是小时但超出范围
                df.at[idx, 'value'] = value % 24
            elif value < 0:
                df.at[idx, 'value'] = 0
        
        # 2. 血氧相关，限制在合理范围 (90-100)
        elif 'blood_oxygen' in indicator or 'oxygen' in indicator:
            if value > 100:
                df.at[idx, 'value'] = min(100, max(90, value % 100 + 90))
            elif value < 50:
                df.at[idx, 'value'] = max(90, 95 + (value % 10))
        
        # 3. 心率相关，限制在合理范围 (40-200)
        elif 'heart_rate' in indicator or 'hr_' in indicator:
            if value > 200:
                df.at[idx, 'value'] = min(200, max(40, value % 160 + 40))
            elif value < 30:
                df.at[idx, 'value'] = max(40, 60 + (value % 20))
        
        # 4. 百分比类型，限制在0-100
        elif any(keyword in indicator for keyword in ['percentage', 'percent', 'ratio']):
            if value > 100:
                df.at[idx, 'value'] = value % 100
            elif value < 0:
                df.at[idx, 'value'] = abs(value) % 100
        
        # 5. 步数相关，限制在合理范围 (0-50000)
        elif 'steps' in indicator:
            if value > 50000:
                df.at[idx, 'value'] = value % 50000
            elif value < 0:
                df.at[idx, 'value'] = abs(value) % 50000
        
        # 6. 时长类型(秒)，转换为分钟并限制合理范围
        elif 'duration' in indicator:
            if value > 86400:  # 超过24小时的秒数
                df.at[idx, 'value'] = min(1440, (value % 86400) / 60)  # 转换为分钟，最大24小时
            elif value > 3600:  # 超过1小时
                df.at[idx, 'value'] = min(1440, value / 60)  # 转换为分钟，最大24小时
            elif value > 1440:  # 如果已经是分钟但超过24小时
                df.at[idx, 'value'] = value % 1440  # 限制在24小时内
            elif value < 0:
                df.at[idx, 'value'] = 0
        
        # 7. 计数类型，确保为合理整数
        elif any(keyword in indicator for keyword in ['count', '_days_', 'frequency']):
            if value > 365:  # 超过一年
                df.at[idx, 'value'] = value % 365
            elif value < 0:
                df.at[idx, 'value'] = 0
            df.at[idx, 'value'] = int(df.at[idx, 'value'])
        
        # 8. 距离相关，限制在合理范围(米)
        elif 'distance' in indicator:
            if value > 100000:  # 超过100公里
                df.at[idx, 'value'] = value % 100000
            elif value < 0:
                df.at[idx, 'value'] = 0
        
        # 9. VO2 max相关，限制在合理范围
        elif 'vo2' in indicator:
            if value > 100:
                df.at[idx, 'value'] = min(80, max(10, value % 80 + 10))
            elif value < 0:
                df.at[idx, 'value'] = max(10, 30 + (abs(value) % 20))
    
    # 确保所有value都是有限数值
    df['value'] = df['value'].fillna(0)
    df['value'] = df['value'].replace([np.inf, -np.inf], 0)
    
    # 统计处理结果
    value_stats = df['value'].describe()
    print(f"📊 Value字段处理完成:")
    print(f"📊   范围: {value_stats['min']:.2f} - {value_stats['max']:.2f}")
    print(f"📊   平均: {value_stats['mean']:.2f}")
    print(f"📊   中位数: {value_stats['50%']:.2f}")
    
    return df

def _optimize_data_types(df):
    """优化数据类型"""
    print("📊 优化数据类型...")
    
    for col in progress_bar(df.columns, desc="优化数据类型"):
        if df[col].dtype == 'object':
            # 尝试转换为数值类型
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if not numeric_data.isna().all():
                df[col] = numeric_data
            else:
                # 尝试转换为datetime
                try:
                    datetime_data = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_data.isna().all():
                        df[col] = datetime_data
                    else:
                        # 转换为category类型（内存优化）
                        if df[col].nunique() < len(df) * 0.5:
                            df[col] = df[col].astype('category')
                except:
                    pass
        
        # 优化整数类型
        elif df[col].dtype in ['int64', 'int32']:
            min_val, max_val = df[col].min(), df[col].max()
            if min_val >= 0 and max_val < 256:
                df[col] = df[col].astype('uint8')
            elif min_val >= -128 and max_val < 128:
                df[col] = df[col].astype('int8')
            elif min_val >= -32768 and max_val < 32768:
                df[col] = df[col].astype('int16')
        
        # 优化浮点类型
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def _reduce_cardinality_smart(df, strategy):
    """智能减少高基数列，保护重要的数值字段"""
    print(f"📊 智能减少高基数列 (策略: {strategy})...")
    
    # 定义不应该进行基数减少的字段
    protected_fields = ['value', 'user_id', 'id']
    
    for col in progress_bar(df.columns, desc="处理高基数列"):
        # 跳过保护的数值字段
        if col in protected_fields:
            continue
            
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_count = df[col].nunique()
            
            # 如果唯一值过多，进行处理
            if unique_count > 50:  # 降低阈值，因为是单用户数据
                # 先转换为object类型以避免category类型的限制
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype('object')
                
                if strategy == 'frequency_based':
                    # 基于频率保留前N个值
                    top_values = df[col].value_counts().head(30).index  # 减少保留数量
                    df.loc[~df[col].isin(top_values), col] = 'other'
                    
                elif strategy == 'adaptive':
                    # 自适应策略：保留覆盖90%数据的值
                    value_counts = df[col].value_counts()
                    cumsum = value_counts.cumsum()
                    threshold = len(df) * 0.9
                    keep_values = value_counts[cumsum <= threshold].index
                    df.loc[~df[col].isin(keep_values), col] = 'other'
                    
                elif strategy == 'simple':
                    # 简单策略：保留前15个最常见的值
                    top_values = df[col].value_counts().head(15).index
                    df.loc[~df[col].isin(top_values), col] = 'other'
                
                new_unique_count = df[col].nunique()
                print(f"📊   {col}: {unique_count} → {new_unique_count} 个唯一值")
    
    return df

def _handle_time_columns(df):
    """智能处理时间列"""
    print("📊 处理时间列...")
    
    time_columns = ['start_time', 'end_time', 'create_time', 'update_time']
    for col in progress_bar(time_columns, desc="处理时间列"):
        if col in df.columns:
            try:
                # 转换为datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # 检查有效数据的比例
                valid_rate = df[col].notna().sum() / len(df)
                if valid_rate < 0.5:
                    # 如果大部分数据无效，删除该列
                    df = df.drop(columns=[col])
                    continue
                
                # 提取时间特征
                if col in ['start_time', 'end_time']:
                    df[f'{col}_year'] = df[col].dt.year.fillna(2023)
                    df[f'{col}_month'] = df[col].dt.month.fillna(1)
                    df[f'{col}_day'] = df[col].dt.day.fillna(1)
                    df[f'{col}_hour'] = df[col].dt.hour.fillna(0)
                    df[f'{col}_weekday'] = df[col].dt.dayofweek.fillna(0)
                
                # 删除原始时间列
                df = df.drop(columns=[col])
                
            except Exception as e:
                print(f"📊   {col} 处理失败: {e}")
                df = df.drop(columns=[col], errors='ignore')
    
    return df

def _handle_outliers_smart(df):
    """智能处理异常值，保护重要字段"""
    print("📊 智能处理异常值...")
    
    # 定义不处理异常值的字段
    protected_fields = ['value', 'user_id', 'id']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in progress_bar(numeric_cols, desc="处理异常值"):
        # 跳过保护的字段
        if col in protected_fields:
            continue
            
        if df[col].std() == 0:  # 跳过常数列
            continue
            
        # 使用IQR方法识别异常值
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # 避免除零错误
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 计算异常值比例
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_ratio = outliers.sum() / len(df)
            
            # 如果异常值比例不太高（<10%），进行处理
            if 0 < outlier_ratio < 0.1:
                # 使用边界值替换异常值
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                print(f"📊   {col}: 处理了 {outliers.sum()} 个异常值 ({outlier_ratio*100:.1f}%)")
    
    return df

def _final_cleanup(df):
    """最终清理"""
    print("📊 最终数据清理...")
    
    # 删除完全空的行
    df = df.dropna(how='all')
    
    # 确保所有category类型转换为object类型（兼容所有SDV模型）
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype('object')
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    return df

def create_metadata(df):
    """创建元数据"""
    print("📊 正在创建元数据...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    print("📊 元数据创建完成")
    return metadata

def preprocess_sequential_data(df, user_col='user_id', target_user_id=169, sample_size=None):
    """专门为序列模型预处理数据"""
    print("📊 开始序列数据预处理...")
    original_shape = df.shape
    
    # 1. 数据清理
    df_processed = df.copy()
    
    # 删除无用列
    columns_to_drop = ['id', 'source_table_id', 'comment', 'indicator_id']
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
    
    # 2. 选择单个用户的数据
    if user_col in df_processed.columns and target_user_id:
        user_data = df_processed[df_processed[user_col] == target_user_id].copy()
        if len(user_data) > 0:
            df_processed = user_data
            print(f"📊 ✓ 选择用户{target_user_id}的数据: {len(df_processed)}条记录")
        else:
            print(f"📊 ⚠️ 未找到用户{target_user_id}的数据，使用所有用户数据")
    
    # 3. 处理时间列
    df_processed = _handle_sequential_time_columns(df_processed)
    
    # 4. 数据采样
    if sample_size and sample_size < len(df_processed):
        df_processed = df_processed.sample(n=sample_size, random_state=42)
        print(f"📊 ✓ 数据采样: {len(df_processed)} 行")
    
    # 5. 保护数值字段（特别是value字段）
    df_processed = _protect_numeric_fields(df_processed)
    
    # 6. 处理高基数列（智能保护）
    df_processed = _reduce_cardinality_smart(df_processed, 'frequency_based')
    
    # 7. 处理缺失值
    df_processed = _handle_missing_values(df_processed)
    
    # 8. 创建序列索引
    df_processed = _create_sequence_index(df_processed, user_col)
    
    # 9. 最终清理
    df_processed = _final_cleanup(df_processed)
    
    print(f"📊 ✓ 序列预处理完成: {original_shape} → {df_processed.shape}")
    
    return df_processed

def _handle_sequential_time_columns(df):
    """处理序列数据的时间列"""
    print("📊 处理序列时间列...")
    
    # 优先使用create_time，因为它的数据完整性更好
    time_columns = ['create_time', 'start_time', 'update_time', 'end_time']
    
    for time_col in time_columns:
        if time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                
                # 检查有效性
                valid_count = df[time_col].notna().sum()
                if valid_count > len(df) * 0.8:  # 如果80%以上的数据有效
                    # 创建连续的序列时间索引
                    df = df.sort_values([time_col]).reset_index(drop=True)
                    df['sequence_datetime'] = df[time_col]
                    
                    # 提取时间特征
                    df['hour'] = df[time_col].dt.hour
                    df['day_of_week'] = df[time_col].dt.dayofweek
                    df['month'] = df[time_col].dt.month
                    
                    print(f"📊 ✓ 使用 {time_col} 创建序列时间索引")
                    break
                    
            except Exception as e:
                print(f"📊 ⚠️ {time_col} 处理失败: {e}")
                continue
    else:
        # 如果没有可用的时间列，创建人工序列索引
        df['sequence_datetime'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
        print("📊 ⚠️ 创建人工序列时间索引")
    
    return df

def _create_sequence_index(df, user_col):
    """创建序列索引"""
    print("📊 创建序列索引...")
    
    if user_col in df.columns:
        # 为每个用户创建独立的序列索引
        def create_user_sequence(group):
            group = group.copy()
            group['sequence_index'] = range(len(group))
            return group
        
        df = df.groupby(user_col).apply(create_user_sequence).reset_index(drop=True)
    else:
        # 创建全局序列索引
        df['sequence_index'] = range(len(df))
    
    return df

def create_sequential_metadata(df):
    """创建序列数据元数据"""
    print("📊 正在创建序列元数据...")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # 设置序列键
    if 'user_id' in df.columns:
        metadata.update_column('user_id', sdtype='id')
        metadata.set_sequence_key('user_id')
        print("📊 ✓ 设置序列键为: user_id")
    
    # 设置序列索引
    sequence_index = None
    for col in ['sequence_datetime', 'sequence_index', 'day_of_year']:
        if col in df.columns:
            if col == 'sequence_datetime':
                metadata.update_column(col, sdtype='datetime')
            else:
                metadata.update_column(col, sdtype='numerical')
            sequence_index = col
            break
    
    if sequence_index:
        metadata.set_sequence_index(sequence_index)
        print(f"📊 ✓ 设置序列索引为: {sequence_index}")
    
    print(f"📊 ✓ 序列元数据创建完成，sequence_key: user_id")
    return metadata

def train_and_sample(synthesizer, real_data, num_rows=1000):
    """训练合成器并生成数据"""
    print("📊 正在训练合成器...")
    start_time = datetime.now()
    
    synthesizer.fit(real_data)
    
    train_time = datetime.now() - start_time
    print(f"📊 训练完成，耗时: {train_time}")
    
    print(f"📊 正在生成 {num_rows} 行合成数据...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    return synthetic_data, train_time

def evaluate_model(real_data, synthetic_data, metadata, model_name):
    """评估模型性能"""
    print(f"📊 正在评估 {model_name} 模型...")
    
    try:
        quality_report = evaluate_quality(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        
        overall_score = quality_report.get_score()
        print(f"📊 {model_name} 总体质量分数: {overall_score*100:.2f}%")
        
        return quality_report
    except Exception as e:
        print(f"📊 评估过程中出现错误: {e}")
        return None

def save_results(synthetic_data, model_name, output_dir, metadata=None, quality_report=None, train_time=None):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合成数据
    output_file = os.path.join(output_dir, f'{model_name}_synthetic_data.csv')
    synthetic_data.to_csv(output_file, index=False)
    print(f"📊 ✓ 合成数据已保存至: {output_file}")
    
    # 生成并保存汇总报告
    summary_file = os.path.join(output_dir, f'{model_name}_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} 模型训练结果汇总 ===\n\n")
        f.write(f"训练时间: {train_time}\n")
        f.write(f"生成数据行数: {len(synthetic_data)}\n")
        f.write(f"生成数据列数: {len(synthetic_data.columns)}\n")
        
        if quality_report:
            overall_score = quality_report.get_score()
            f.write(f"总体质量分数: {overall_score*100:.2f}%\n")
            
            # 质量等级评定
            if overall_score >= 0.9:
                quality_level = "卓越"
            elif overall_score >= 0.8:
                quality_level = "优秀"
            elif overall_score >= 0.7:
                quality_level = "良好"
            elif overall_score >= 0.6:
                quality_level = "一般"
            else:
                quality_level = "需要改进"
            
            f.write(f"质量等级: {quality_level}\n")
        
        f.write(f"\n数据样例:\n")
        f.write(synthetic_data.head().to_string())
        
        f.write(f"\n\n数据统计:\n")
        f.write(synthetic_data.describe().to_string())
    
    print(f"📊 ✓ 汇总报告已保存至: {summary_file}")
    return summary_file 