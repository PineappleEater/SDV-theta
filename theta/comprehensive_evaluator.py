#!/usr/bin/env python3
"""
综合数据评测器 - 专注于分析和报告生成
功能：
- 读取已生成的合成数据
- 统计分析与原数据对比
- 生成分析图表
- 生成完整分析报告
注意：模型训练和数据生成请使用 run_all_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端和样式
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.style.use('default')

class ComprehensiveEvaluator:
    def __init__(self, data_path='source_data/th_series_data.csv'):
        """初始化评测器"""
        self.data_path = data_path
        self.original_data = None
        self.user_data = None
        self.synthetic_datasets = {}  # 存储多个模型的合成数据
        self.user_id = None
        self.evaluation_results = {}
        
        # 支持的模型列表
        self.supported_models = [
            'gaussian_copula',
            'ctgan', 
            'copulagan',
            'tvae',
            'par'
        ]
        
        # 创建输出目录
        os.makedirs('output/graph', exist_ok=True)
        os.makedirs('output/comprehensive_reports', exist_ok=True)
    
    def load_data(self):
        """加载原始数据"""
        print("📂 加载原始数据...")
        self.original_data = pd.read_csv(self.data_path)
        print(f"✓ 数据加载完成: {self.original_data.shape}")
        return True
    
    def analyze_users(self, min_records=200, max_records=5000):
        """分析可用用户"""
        print(f"\n👥 分析可用用户 (记录数范围: {min_records}-{max_records})...")
        
        user_counts = self.original_data['user_id'].value_counts()
        suitable_users = user_counts[(user_counts >= min_records) & (user_counts <= max_records)]
        
        print(f"📊 符合条件的用户: {len(suitable_users)}个")
        
        if len(suitable_users) == 0:
            print("❌ 没有找到合适的用户")
            return None
        
        # 显示推荐用户
        print("\n推荐用户:")
        for i, (user_id, count) in enumerate(suitable_users.head(10).items(), 1):
            user_data = self.original_data[self.original_data['user_id'] == user_id]
            indicators = user_data['indicator'].nunique()
            
            try:
                dates = pd.to_datetime(user_data['start_time'], errors='coerce').dropna()
                date_range = (dates.max() - dates.min()).days if len(dates) > 0 else 0
            except:
                date_range = 0
            
            print(f"  {i:2d}. 用户{user_id}: {count}条记录, {indicators}种指标, {date_range}天")
        
        return suitable_users
    
    def select_user(self, user_id, max_records=None):
        """选择并准备用户数据"""
        print(f"\n🎯 选择用户 {user_id}...")
        
        if user_id not in self.original_data['user_id'].values:
            print(f"❌ 用户 {user_id} 不存在")
            return False
        
        self.user_id = user_id
        
        # 提取用户数据
        self.user_data = self.original_data[self.original_data['user_id'] == user_id].copy()
        print(f"✓ 原始数据: {len(self.user_data)} 条记录")
        
        # 数据采样（如果指定了max_records）
        if max_records and len(self.user_data) > max_records:
            self.user_data = self.user_data.sample(n=max_records, random_state=42)
            print(f"✓ 采样至: {len(self.user_data)} 条记录")
        else:
            print(f"✓ 使用全部数据: {len(self.user_data)} 条记录")
        
        # 数据预处理
        self._preprocess_user_data()
        
        return True
    
    def _preprocess_user_data(self):
        """预处理用户数据"""
        print("🔧 预处理用户数据...")
        
        # 删除不需要的列
        columns_to_drop = ['id', 'source_table_id', 'comment', 'indicator_id', 'user_id']
        for col in columns_to_drop:
            if col in self.user_data.columns:
                self.user_data = self.user_data.drop(columns=[col])
        
        # 处理时间 - 优先使用start_time，如果无效则使用create_time
        time_column = None
        if 'start_time' in self.user_data.columns:
            try:
                start_time = pd.to_datetime(self.user_data['start_time'], errors='coerce')
                valid_start_time = start_time.notna().sum()
                
                if valid_start_time > len(self.user_data) * 0.5:  # 如果超过50%的数据有效
                    time_column = 'start_time'
                    self.user_data['start_time'] = start_time
                    self.user_data = self.user_data.dropna(subset=['start_time'])
                    
                    # 按时间排序并创建连续日期时间序列
                    self.user_data = self.user_data.sort_values('start_time').reset_index(drop=True)
                    self.last_date = self.user_data['start_time'].max()
                    
                    # 创建连续的日期时间序列
                    base_time = self.user_data['start_time'].min()
                    self.user_data['sequence_datetime'] = pd.to_datetime([
                        base_time + pd.Timedelta(hours=i) for i in range(len(self.user_data))
                    ])
                    
                    self.user_data['hour'] = self.user_data['start_time'].dt.hour
                    
                    # 保留原始时间用于参考
                    self.user_data['original_start_time'] = self.user_data['start_time']
                    self.user_data = self.user_data.drop(columns=['start_time'])
                    print(f"✓ 使用start_time进行时间处理，创建datetime序列 {self.user_data['sequence_datetime'].min()} 到 {self.user_data['sequence_datetime'].max()}，最后日期: {self.last_date.date()}")
                else:
                    print(f"⚠️ start_time有效率较低 ({valid_start_time}/{len(self.user_data)})，尝试使用create_time")
                    time_column = 'create_time'
                    
            except Exception as e:
                print(f"⚠️ start_time处理失败: {e}，尝试使用create_time")
                time_column = 'create_time'
        
        # 如果start_time不可用，使用create_time
        if time_column != 'start_time' and 'create_time' in self.user_data.columns:
            try:
                self.user_data['create_time'] = pd.to_datetime(self.user_data['create_time'], errors='coerce')
                valid_create_time = self.user_data['create_time'].notna().sum()
                
                if valid_create_time > 0:
                    self.user_data = self.user_data.dropna(subset=['create_time'])
                    self.last_date = self.user_data['create_time'].max()
                    
                    # 对于序列模型，创建基于时间的连续序列
                    # 按create_time排序
                    self.user_data = self.user_data.sort_values('create_time').reset_index(drop=True)
                    
                    # 创建连续的日期时间序列（每条记录间隔1小时）
                    base_time = self.user_data['create_time'].min()
                    self.user_data['sequence_datetime'] = pd.to_datetime([
                        base_time + pd.Timedelta(hours=i) for i in range(len(self.user_data))
                    ])
                    
                    self.user_data['hour'] = self.user_data['create_time'].dt.hour
                    
                    # 保留原始create_time用于参考
                    self.user_data['original_create_time'] = self.user_data['create_time']
                    self.user_data = self.user_data.drop(columns=['create_time'])
                    
                    print(f"✓ 使用create_time进行时间处理，创建datetime序列 {self.user_data['sequence_datetime'].min()} 到 {self.user_data['sequence_datetime'].max()}，最后日期: {self.last_date.date()}")
                else:
                    print(f"⚠️ create_time也无效，使用当前时间")
                    self.last_date = datetime.now()
                    
            except Exception as e:
                print(f"⚠️ create_time处理失败: {e}")
                self.last_date = datetime.now()
        
        if time_column is None:
            self.last_date = datetime.now()
        
        # 处理value列 - 应用智能处理逻辑
        if 'value' in self.user_data.columns:
            # 应用与训练数据相同的智能value处理
            self.user_data = self._smart_process_value_field(self.user_data)
            print(f"✓ value智能处理完成")
        
        # 处理indicator列 - 为了与训练数据保持一致，不进行'other'标记
        if 'indicator' in self.user_data.columns:
            # 保持原始指标，不进行'other'标记以与训练数据一致
            indicator_count = self.user_data['indicator'].nunique()
            print(f"✓ 保持原始指标 (共{indicator_count}个)，与训练数据保持一致")
        
        # 排序
        if 'day_of_year' in self.user_data.columns:
            self.user_data = self.user_data.sort_values('day_of_year').reset_index(drop=True)
        
        print(f"✓ 预处理完成: {self.user_data.shape}")
    
    def load_synthetic_data(self, models=None):
        """加载已生成的合成数据"""
        print(f"\n📂 加载合成数据...")
        
        if models is None:
            models = self.supported_models
        
        loaded_models = []
        
        for model in models:
            # 构建数据文件路径 - 支持多种文件命名格式
            data_files = [
                f"output/{model}/{model.upper()}_synthetic_data.csv",
                f"output/{model}/{model.title()}_synthetic_data.csv", 
                f"output/{model}/Enhanced_{model.upper()}_synthetic_data.csv",
                f"output/{model}/synthetic_data.csv"
            ]
            
            # 针对特殊情况的额外文件名
            if model == 'gaussian_copula':
                data_files.extend([
                    "output/gaussian_copula/GaussianCopula_synthetic_data.csv",
                    "output/gaussian_copula/Gaussian_Copula_synthetic_data.csv"
                ])
            elif model == 'par':
                data_files.extend([
                    "output/par_enhanced/Enhanced_PAR_synthetic_data.csv",
                    "output/par/Enhanced_PAR_synthetic_data.csv"
                ])
            
            # 尝试多种可能的文件名
            loaded = False
            for data_file in data_files:
                if os.path.exists(data_file):
                    try:
                        synthetic_data = pd.read_csv(data_file)
                        
                        # 过滤用户数据（如果包含user_id列）
                        if 'user_id' in synthetic_data.columns and self.user_id:
                            user_synthetic = synthetic_data[synthetic_data['user_id'] == self.user_id]
                            if len(user_synthetic) > 0:
                                synthetic_data = user_synthetic
                            # 如果没有用户数据，使用全部数据
                        
                        # 处理时间列：如果没有hour列但有start_time_hour列，则创建hour列
                        if 'hour' not in synthetic_data.columns and 'start_time_hour' in synthetic_data.columns:
                            synthetic_data['hour'] = synthetic_data['start_time_hour']
                        
                        self.synthetic_datasets[model] = synthetic_data
                        
                        print(f"✅ {model}: 加载 {len(self.synthetic_datasets[model])} 条数据")
                        loaded_models.append(model)
                        loaded = True
                        break
                        
                    except Exception as e:
                        print(f"⚠️ {model}: 读取文件失败 - {e}")
                        continue
            
            if not loaded:
                print(f"❌ {model}: 未找到合成数据文件")
                # 尝试搜索输出目录
                search_pattern = f"output/{model}/*synthetic*.csv"
                found_files = glob.glob(search_pattern)
                if found_files:
                    print(f"   💡 发现文件: {found_files}")
        
        if not loaded_models:
            print("❌ 未找到任何合成数据文件")
            print("💡 请先运行 python3 run_all_models.py 生成合成数据")
            return False
        
        print(f"✅ 成功加载 {len(loaded_models)} 个模型的合成数据: {', '.join(loaded_models)}")
        return loaded_models
    
    def statistical_analysis(self):
        """统计分析 - 支持多模型对比"""
        print(f"\n📊 进行统计分析...")
        
        if not self.synthetic_datasets:
            print("❌ 没有合成数据可供分析")
            return {}
        
        results = {}
        
        # 基本统计
        original_records = len(self.user_data)
        original_indicators = self.user_data['indicator'].nunique() if 'indicator' in self.user_data.columns else 0
        
        results['basic_stats'] = {
            'original_records': original_records,
            'original_indicators': original_indicators,
        }
        
        # 为每个模型添加统计
        for model_name, synthetic_data in self.synthetic_datasets.items():
            results['basic_stats'][f'{model_name}_records'] = len(synthetic_data)
            results['basic_stats'][f'{model_name}_indicators'] = synthetic_data['indicator'].nunique() if 'indicator' in synthetic_data.columns else 0
        
        # 指标分布对比 - 支持多模型
        if 'indicator' in self.user_data.columns:
            orig_indicators = self.user_data['indicator'].value_counts()
            
            results['indicator_analysis'] = {
                'original_top5': orig_indicators.head().to_dict(),
                'models': {}
            }
            
            # 详细分析前16个最常见指标
            top16_indicators = orig_indicators.head(16)
            
            # 为每个模型分析指标
            for model_name, synthetic_data in self.synthetic_datasets.items():
                if 'indicator' not in synthetic_data.columns:
                    continue
                    
                synt_indicators = synthetic_data['indicator'].value_counts()
                
                # 前16个指标的详细分析
                top16_analysis = {}
                for indicator in top16_indicators.index:
                    orig_count = orig_indicators.get(indicator, 0)
                    synt_count = synt_indicators.get(indicator, 0)
                    
                    # 计算该指标的数值分布相似性
                    indicator_similarity = self._analyze_indicator_values(indicator, orig_count, synt_count, synthetic_data)
                    
                    top16_analysis[indicator] = {
                        'original_count': orig_count,
                        'synthetic_count': synt_count,
                        'count_similarity': 1 - abs(orig_count - synt_count) / max(orig_count, synt_count, 1),
                        'value_analysis': indicator_similarity
                    }
                
                results['indicator_analysis']['models'][model_name] = {
                    'synthetic_all_indicators': synt_indicators.to_dict(),  # 完整指标统计，用于图表生成
                    'synthetic_top10': synt_indicators.head(10).to_dict(),  # 改为前10个指标
                    'synthetic_top5': synt_indicators.head(5).to_dict(),   # 保持向后兼容
                    'coverage_similarity': self._calculate_coverage_similarity(orig_indicators, synt_indicators),
                    'top16_detailed_analysis': top16_analysis
                }
        
        # 数值分布对比 - 支持多模型
        if 'value' in self.user_data.columns:
            try:
                orig_values = pd.to_numeric(self.user_data['value'], errors='coerce').dropna()
                
                if len(orig_values) > 0:
                    results['value_analysis'] = {
                        'original_stats': orig_values.describe().to_dict(),
                        'models': {}
                    }
                    
                    for model_name, synthetic_data in self.synthetic_datasets.items():
                        if 'value' in synthetic_data.columns:
                            synt_values = pd.to_numeric(synthetic_data['value'], errors='coerce').dropna()
                            
                            if len(synt_values) > 0:
                                results['value_analysis']['models'][model_name] = {
                                    'synthetic_stats': synt_values.describe().to_dict(),
                                    'distribution_similarity': self._calculate_distribution_similarity(orig_values, synt_values)
                                }
            except:
                results['value_analysis'] = {'error': 'Failed to analyze numeric values'}
        
        # 时间分布对比 - 支持多模型
        if 'hour' in self.user_data.columns:
            orig_hours = self.user_data['hour'].value_counts().sort_index()
            
            results['time_analysis'] = {
                'original_hour_dist': orig_hours.to_dict(),
                'models': {}
            }
            
            for model_name, synthetic_data in self.synthetic_datasets.items():
                if 'hour' in synthetic_data.columns:
                    synt_hours = synthetic_data['hour'].value_counts().sort_index()
                    
                    results['time_analysis']['models'][model_name] = {
                        'synthetic_hour_dist': synt_hours.to_dict(),
                        'peak_hours_similarity': self._calculate_peak_similarity(orig_hours, synt_hours)
                    }
        
        self.evaluation_results = results
        print(f"✅ 统计分析完成")
        
        return results
    
    def _smart_process_value_field(self, df):
        """根据indicator类型智能处理value字段（与训练时保持一致）"""
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
        timestamp_count = 0
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
                        timestamp_count += 1
                        if timestamp_count <= 10:  # 只显示前10个转换
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
        
        if timestamp_count > 0:
            print(f"📊   共转换了 {timestamp_count} 个时间戳")
        
        return df
    
    def _calculate_coverage_similarity(self, orig_dist, synt_dist):
        """计算覆盖率相似性"""
        common_items = set(orig_dist.index) & set(synt_dist.index)
        total_items = set(orig_dist.index) | set(synt_dist.index)
        return len(common_items) / len(total_items) if total_items else 0
    
    def _calculate_distribution_similarity(self, orig_values, synt_values):
        """计算分布相似性"""
        try:
            # 使用KS检验统计量的相似性
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(orig_values, synt_values)
            return 1 - ks_stat  # 转换为相似性分数
        except:
            # 如果scipy不可用，使用简单的统计量对比
            orig_mean, orig_std = orig_values.mean(), orig_values.std()
            synt_mean, synt_std = synt_values.mean(), synt_values.std()
            
            mean_diff = abs(orig_mean - synt_mean) / max(orig_mean, synt_mean, 1)
            std_diff = abs(orig_std - synt_std) / max(orig_std, synt_std, 1)
            
            return 1 - (mean_diff + std_diff) / 2
    
    def _calculate_peak_similarity(self, orig_hours, synt_hours):
        """计算峰值时间相似性"""
        orig_peak = orig_hours.idxmax()
        synt_peak = synt_hours.idxmax()
        
        # 计算峰值时间差异
        time_diff = abs(orig_peak - synt_peak)
        time_diff = min(time_diff, 24 - time_diff)  # 考虑循环性
        
        return 1 - time_diff / 12  # 标准化到0-1
    
    def _analyze_indicator_values(self, indicator, orig_count, synt_count, synthetic_data):
        """分析特定指标的数值分布"""
        try:
            # 提取该指标的原始数据值
            orig_indicator_data = self.user_data[self.user_data['indicator'] == indicator]['value']
            synt_indicator_data = synthetic_data[synthetic_data['indicator'] == indicator]['value']
            
            if len(orig_indicator_data) == 0 or len(synt_indicator_data) == 0:
                return {'error': 'No data for this indicator'}
            
            # 尝试数值化分析
            orig_numeric = pd.to_numeric(orig_indicator_data, errors='coerce')
            synt_numeric = pd.to_numeric(synt_indicator_data, errors='coerce')
            
            orig_numeric_valid = orig_numeric.dropna()
            synt_numeric_valid = synt_numeric.dropna()
            
            analysis = {}
            
            # 如果数值化成功率高，进行数值分析
            if len(orig_numeric_valid) > len(orig_indicator_data) * 0.7 and len(synt_numeric_valid) > len(synt_indicator_data) * 0.7:
                analysis['type'] = 'numerical'
                analysis['original_stats'] = {
                    'mean': float(orig_numeric_valid.mean()),
                    'std': float(orig_numeric_valid.std()),
                    'min': float(orig_numeric_valid.min()),
                    'max': float(orig_numeric_valid.max()),
                    'count': len(orig_numeric_valid)
                }
                analysis['synthetic_stats'] = {
                    'mean': float(synt_numeric_valid.mean()),
                    'std': float(synt_numeric_valid.std()),
                    'min': float(synt_numeric_valid.min()),
                    'max': float(synt_numeric_valid.max()),
                    'count': len(synt_numeric_valid)
                }
                
                # 计算数值分布相似性
                if len(orig_numeric_valid) > 0 and len(synt_numeric_valid) > 0:
                    analysis['distribution_similarity'] = self._calculate_distribution_similarity(orig_numeric_valid, synt_numeric_valid)
                else:
                    analysis['distribution_similarity'] = 0.0
            else:
                # 分类数据分析
                analysis['type'] = 'categorical'
                orig_values = orig_indicator_data.value_counts()
                synt_values = synt_indicator_data.value_counts()
                
                analysis['original_top_values'] = orig_values.head(5).to_dict()
                analysis['synthetic_top_values'] = synt_values.head(5).to_dict()
                analysis['value_similarity'] = self._calculate_coverage_similarity(orig_values, synt_values)
            
            return analysis
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _generate_top16_indicators_chart(self):
        """生成前16个指标的详细分析图表 - 多模型支持"""
        print(f"📊 生成前16个指标详细分析图表...")
        
        try:
            indicator_analysis = self.evaluation_results.get('indicator_analysis', {})
            if 'models' not in indicator_analysis or not indicator_analysis['models']:
                print("⚠️ 前16个指标分析数据不可用")
                return None
            
            # 选择第一个可用模型进行详细展示
            first_model = list(indicator_analysis['models'].keys())[0]
            top16_data = indicator_analysis['models'][first_model]['top16_detailed_analysis']
            synthetic_data = self.synthetic_datasets[first_model]
            
            # 创建4x4的子图
            fig, axes = plt.subplots(4, 4, figsize=(24, 20))
            fig.suptitle(f'User {self.user_id} - Top 16 Indicators Detailed Analysis', fontsize=18, fontweight='bold')
            
            # 获取所有指标名称
            indicators = list(top16_data.keys())[:16]  # 确保最多16个
            
            for i, indicator in enumerate(indicators):
                row = i // 4
                col = i % 4
                ax = axes[row, col]
                
                indicator_info = top16_data[indicator]
                value_analysis = indicator_info.get('value_analysis', {})
                
                # 根据数据类型选择不同的可视化方式
                if value_analysis.get('type') == 'numerical':
                    # 数值型数据：直方图对比
                    self._plot_numerical_indicator(ax, indicator, value_analysis, synthetic_data)
                elif value_analysis.get('type') == 'categorical':
                    # 分类型数据：条形图对比
                    self._plot_categorical_indicator(ax, indicator, value_analysis, synthetic_data)
                else:
                    # 如果分析失败，显示基本统计
                    self._plot_basic_indicator_stats(ax, indicator, indicator_info)
                
                # 设置标题（截断过长的指标名）
                title = indicator if len(indicator) <= 15 else indicator[:12] + "..."
                ax.set_title(title, fontsize=10, fontweight='bold')
            
            # 隐藏多余的子图
            for i in range(len(indicators), 16):
                row = i // 4
                col = i % 4
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图表
            top16_chart_file = f'output/graph/user_{self.user_id}_top16_indicators_analysis.png'
            plt.savefig(top16_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 前16个指标分析图表已保存: {top16_chart_file}")
            return top16_chart_file
            
        except Exception as e:
            print(f"⚠️ 前16个指标图表生成失败: {e}")
            return None
    
    def _plot_numerical_indicator(self, ax, indicator, value_analysis, synthetic_data):
        """绘制数值型指标的分布对比"""
        try:
            # 提取该指标的数据
            orig_data = self.user_data[self.user_data['indicator'] == indicator]['value']
            synt_data = synthetic_data[synthetic_data['indicator'] == indicator]['value']
            
            orig_numeric = pd.to_numeric(orig_data, errors='coerce').dropna()
            synt_numeric = pd.to_numeric(synt_data, errors='coerce').dropna()
            
            if len(orig_numeric) > 0 and len(synt_numeric) > 0:
                # 创建直方图
                ax.hist(orig_numeric, bins=min(20, max(5, len(orig_numeric)//10)), 
                       alpha=0.6, label='Original', density=True, color='skyblue')
                ax.hist(synt_numeric, bins=min(20, max(5, len(synt_numeric)//10)), 
                       alpha=0.6, label='Synthetic', density=True, color='lightcoral')
                
                # 添加统计信息
                orig_mean = orig_numeric.mean()
                synt_mean = synt_numeric.mean()
                similarity = value_analysis.get('distribution_similarity', 0) * 100
                
                ax.axvline(orig_mean, color='blue', linestyle='--', alpha=0.8, label=f'Orig Mean: {orig_mean:.2f}')
                ax.axvline(synt_mean, color='red', linestyle='--', alpha=0.8, label=f'Synt Mean: {synt_mean:.2f}')
                
                ax.legend(fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                ax.text(0.02, 0.98, f'Similarity: {similarity:.1f}%', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No valid numeric data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes, fontsize=8)
    
    def _plot_categorical_indicator(self, ax, indicator, value_analysis, synthetic_data):
        """绘制分类型指标的分布对比"""
        try:
            orig_values = value_analysis.get('original_top_values', {})
            synt_values = value_analysis.get('synthetic_top_values', {})
            
            if orig_values:
                # 获取前5个最常见的值
                top_values = list(orig_values.keys())[:5]
                
                orig_counts = [orig_values.get(val, 0) for val in top_values]
                synt_counts = [synt_values.get(val, 0) for val in top_values]
                
                x_pos = np.arange(len(top_values))
                width = 0.35
                
                ax.bar(x_pos - width/2, orig_counts, width, label='Original', alpha=0.8, color='skyblue')
                ax.bar(x_pos + width/2, synt_counts, width, label='Synthetic', alpha=0.8, color='lightcoral')
                
                # 设置x轴标签
                labels = [str(val)[:8] + '...' if len(str(val)) > 8 else str(val) for val in top_values]
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, fontsize=8)
                
                ax.legend(fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                
                similarity = value_analysis.get('value_similarity', 0) * 100
                ax.text(0.02, 0.98, f'Similarity: {similarity:.1f}%', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No categorical data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes, fontsize=8)
    
    def _plot_basic_indicator_stats(self, ax, indicator, indicator_info):
        """绘制基本指标统计"""
        try:
            orig_count = indicator_info.get('original_count', 0)
            synt_count = indicator_info.get('synthetic_count', 0)
            count_similarity = indicator_info.get('count_similarity', 0) * 100
            
            categories = ['Original', 'Synthetic']
            counts = [orig_count, synt_count]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.8)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Count', fontsize=8)
            ax.text(0.02, 0.98, f'Count Similarity: {count_similarity:.1f}%', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                   
        except Exception as e:
            ax.text(0.5, 0.5, f'Basic plot error: {str(e)[:20]}...', ha='center', va='center', transform=ax.transAxes, fontsize=8)
    
    def _generate_multi_model_comparison_charts(self):
        """生成多模型对比柱状图"""
        print(f"📊 生成多模型对比图表...")
        
        try:
            charts = []
            
            # 1. 生成指标覆盖率对比图
            coverage_chart = self._generate_coverage_comparison_chart()
            if coverage_chart:
                charts.append(coverage_chart)
            
            # 2. 生成前10个指标分布对比图
            top10_chart = self._generate_top10_indicators_comparison_chart()
            if top10_chart:
                charts.append(top10_chart)
            
            # 3. 生成数值统计对比图
            stats_chart = self._generate_statistics_comparison_chart()
            if stats_chart:
                charts.append(stats_chart)
            
            # 4. 生成时间模式对比图
            time_chart = self._generate_time_pattern_comparison_chart()
            if time_chart:
                charts.append(time_chart)
            
            print(f"✅ 生成了 {len(charts)} 个多模型对比图表")
            return charts
            
        except Exception as e:
            print(f"⚠️ 多模型对比图表生成失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_coverage_comparison_chart(self):
        """生成指标覆盖率对比柱状图"""
        try:
            if 'indicator_analysis' not in self.evaluation_results:
                return None
            
            indicator_analysis = self.evaluation_results['indicator_analysis']
            if 'models' not in indicator_analysis:
                return None
            
            # 创建覆盖率对比图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            models = list(indicator_analysis['models'].keys())
            coverage_rates = []
            
            for model in models:
                model_analysis = indicator_analysis['models'][model]
                coverage_rate = model_analysis.get('coverage_similarity', 0) * 100
                coverage_rates.append(coverage_rate)
            
            bars = ax.bar(models, coverage_rates, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(models)])
            
            # 添加数值标签
            for bar, rate in zip(bars, coverage_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Models Indicator Coverage Similarity Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Coverage Similarity (%)', fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_file = f'output/graph/user_{self.user_id}_coverage_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            print(f"⚠️ 覆盖率对比图生成失败: {e}")
            return None
    
    def _generate_top10_indicators_comparison_chart(self):
        """生成前10个指标分布对比图"""
        try:
            if 'indicator_analysis' not in self.evaluation_results:
                return None
            
            indicator_analysis = self.evaluation_results['indicator_analysis']
            if 'models' not in indicator_analysis:
                return None
            
            # 获取原始数据前10个指标
            orig_top10 = self.user_data['indicator'].value_counts().head(10)
            orig_total = len(self.user_data)
            
            # 创建对比图
            fig, ax = plt.subplots(figsize=(16, 10))
            
            models = list(indicator_analysis['models'].keys())
            x = np.arange(len(orig_top10))
            width = 0.15
            
            # 绘制原始数据 - 转换为频率百分比
            orig_frequencies = (orig_top10.values / orig_total * 100)
            ax.bar(x - width*2, orig_frequencies, width, label='Original', color='darkblue', alpha=0.8)
            
            # 为每个模型绘制柱状图
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
            for i, model in enumerate(models):
                model_analysis = indicator_analysis['models'][model]
                synthetic_all_indicators = model_analysis.get('synthetic_all_indicators', {})
                
                # 获取该模型的总数据量
                model_total = sum(synthetic_all_indicators.values())
                
                # 匹配原始数据的指标顺序 - 从完整指标统计中获取并转换为频率百分比
                model_counts = [synthetic_all_indicators.get(indicator, 0) for indicator in orig_top10.index]
                model_frequencies = [(count / model_total * 100) if model_total > 0 else 0 for count in model_counts]
                
                ax.bar(x + width * (i - 1), model_frequencies, width, 
                      label=model.upper(), color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_title('Top 10 Indicators Frequency Distribution Comparison Across Models', fontsize=14, fontweight='bold')
            ax.set_xlabel('Indicators', fontsize=12)
            ax.set_ylabel('Frequency (%)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([ind[:15] + '...' if len(ind) > 15 else ind for ind in orig_top10.index], 
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            chart_file = f'output/graph/user_{self.user_id}_top10_indicators_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            print(f"⚠️ 前10个指标对比图生成失败: {e}")
            return None
    
    def _generate_statistics_comparison_chart(self):
        """生成数值统计对比图"""
        try:
            if 'value_analysis' not in self.evaluation_results:
                return None
            
            value_analysis = self.evaluation_results['value_analysis']
            if 'models' not in value_analysis or 'original_stats' not in value_analysis:
                return None
            
            orig_stats = value_analysis['original_stats']
            models = list(value_analysis['models'].keys())
            
            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Statistical Measures Comparison Across Models', fontsize=16, fontweight='bold')
            
            stats = ['mean', 'std', 'min', 'max']
            stat_titles = ['Mean Values', 'Standard Deviation', 'Minimum Values', 'Maximum Values']
            
            for idx, (stat, title) in enumerate(zip(stats, stat_titles)):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                if stat in orig_stats:
                    values = [orig_stats[stat]]
                    labels = ['Original']
                    colors = ['darkblue']
                    
                    for model in models:
                        model_stats = value_analysis['models'][model].get('synthetic_stats', {})
                        if stat in model_stats:
                            values.append(model_stats[stat])
                            labels.append(model.upper())
                            colors.append(['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][len(values)-2])
                    
                    bars = ax.bar(labels, values, color=colors, alpha=0.8)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_title(title, fontweight='bold')
                    ax.set_ylabel(stat.title())
                    ax.grid(axis='y', alpha=0.3)
                    
                    if len(labels) > 3:
                        ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_file = f'output/graph/user_{self.user_id}_statistics_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            print(f"⚠️ 统计对比图生成失败: {e}")
            return None
    
    def _generate_time_pattern_comparison_chart(self):
        """生成时间模式对比图"""
        try:
            if 'time_analysis' not in self.evaluation_results:
                return None
            
            time_analysis = self.evaluation_results['time_analysis']
            if 'models' not in time_analysis or 'original_hour_dist' not in time_analysis:
                return None
            
            # 创建时间模式对比图
            fig, ax = plt.subplots(figsize=(14, 8))
            
            orig_hours = time_analysis['original_hour_dist']
            hours = list(range(24))
            orig_counts = [orig_hours.get(h, 0) for h in hours]
            
            # 绘制原始数据
            ax.plot(hours, orig_counts, 'o-', linewidth=3, markersize=8, 
                   label='Original', color='darkblue')
            
            # 为每个模型绘制线图
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
            markers = ['s', '^', 'D', 'v', 'p']
            
            for i, (model, model_time_analysis) in enumerate(time_analysis['models'].items()):
                if 'synthetic_hour_dist' in model_time_analysis:
                    synt_hours = model_time_analysis['synthetic_hour_dist']
                    synt_counts = [synt_hours.get(h, 0) for h in hours]
                    
                    ax.plot(hours, synt_counts, marker=markers[i % len(markers)], 
                           linewidth=2, markersize=6, label=model.upper(), 
                           color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_title('Hourly Activity Pattern Comparison Across Models', fontsize=14, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Activity Count', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            
            plt.tight_layout()
            
            chart_file = f'output/graph/user_{self.user_id}_time_pattern_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            print(f"⚠️ 时间模式对比图生成失败: {e}")
            return None
    
    def generate_visualizations(self):
        """生成可视化图表"""
        print(f"\n📈 生成分析图表...")
        
        try:
            # 获取第一个可用模型用于主要可视化
            if not self.synthetic_datasets:
                print("⚠️ 没有合成数据可用于可视化")
                return None
            
            first_model = list(self.synthetic_datasets.keys())[0]
            first_synthetic_data = self.synthetic_datasets[first_model]
            
            # 创建综合分析图表
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'User {self.user_id} Comprehensive Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. 指标分布对比 - 使用频率百分比
            if 'indicator' in self.user_data.columns:
                orig_indicators = self.user_data['indicator'].value_counts().head(10)
                synt_indicators = first_synthetic_data['indicator'].value_counts() # 使用完整指标统计
                
                # 计算频率百分比
                orig_total = len(self.user_data)
                synt_total = len(first_synthetic_data)
                
                x_pos = np.arange(len(orig_indicators))
                orig_frequencies = (orig_indicators.values / orig_total * 100)
                axes[0, 0].bar(x_pos - 0.2, orig_frequencies, 0.4, label='Original', alpha=0.8)
                
                # 匹配合成数据的指标 - 转换为频率百分比
                synt_matched = [synt_indicators.get(ind, 0) for ind in orig_indicators.index]
                synt_frequencies = [(count / synt_total * 100) if synt_total > 0 else 0 for count in synt_matched]
                axes[0, 0].bar(x_pos + 0.2, synt_frequencies, 0.4, label='Synthetic', alpha=0.8)
                
                axes[0, 0].set_title('Top 10 Indicators Frequency Distribution')
                axes[0, 0].set_xlabel('Indicators')
                axes[0, 0].set_ylabel('Frequency (%)')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 数值分布对比
            if 'value' in self.user_data.columns:
                try:
                    orig_values = pd.to_numeric(self.user_data['value'], errors='coerce').dropna()
                    synt_values = pd.to_numeric(first_synthetic_data['value'], errors='coerce').dropna() # 使用第一个模型的合成数据
                    
                    if len(orig_values) > 0 and len(synt_values) > 0:
                        axes[0, 1].hist(orig_values, bins=30, alpha=0.7, label='Original', density=True)
                        axes[0, 1].hist(synt_values, bins=30, alpha=0.7, label='Synthetic', density=True)
                        axes[0, 1].set_title('Value Distribution Comparison')
                        axes[0, 1].set_xlabel('Value')
                        axes[0, 1].set_ylabel('Density')
                        axes[0, 1].legend()
                except:
                    axes[0, 1].text(0.5, 0.5, 'Numeric Value\nAnalysis Failed', ha='center', va='center')
                    axes[0, 1].set_title('Value Distribution')
            
            # 3. 时间分布对比
            if 'hour' in self.user_data.columns:
                orig_hours = self.user_data['hour'].value_counts().sort_index()
                synt_hours = first_synthetic_data['hour'].value_counts().sort_index() # 使用第一个模型的合成数据
                
                hours = list(range(24))
                orig_counts = [orig_hours.get(h, 0) for h in hours]
                synt_counts = [synt_hours.get(h, 0) for h in hours]
                
                axes[0, 2].plot(hours, orig_counts, 'o-', label='Original', linewidth=2)
                axes[0, 2].plot(hours, synt_counts, 's-', label='Synthetic', linewidth=2)
                axes[0, 2].set_title('Hourly Activity Pattern')
                axes[0, 2].set_xlabel('Hour of Day')
                axes[0, 2].set_ylabel('Activity Count')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 数据质量对比 - 更有意义的相对指标对比
            categories = ['Records\n(K)', 'Indicators\nCoverage (%)', 'Value Types\nCoverage (%)']
            
            # 计算相对指标
            orig_indicators_count = self.user_data['indicator'].nunique()
            synt_indicators_count = first_synthetic_data['indicator'].nunique()
            indicator_coverage = min(synt_indicators_count / orig_indicators_count * 100, 100) if orig_indicators_count > 0 else 0
            
            orig_values_count = self.user_data['value'].nunique() if 'value' in self.user_data.columns else 1
            synt_values_count = first_synthetic_data['value'].nunique() if 'value' in first_synthetic_data.columns else 0
            value_coverage = min(synt_values_count / orig_values_count * 100, 100) if orig_values_count > 0 else 0
            
            # 数据量用千为单位，其他用百分比
            original_metrics = [
                len(self.user_data) / 1000,  # 转换为千
                100,  # 原始数据指标覆盖率为100%
                100   # 原始数据值类型覆盖率为100%
            ]
            synthetic_metrics = [
                len(first_synthetic_data) / 1000,  # 转换为千
                indicator_coverage,
                value_coverage
            ]
            
            x_pos = np.arange(len(categories))
            bars1 = axes[1, 0].bar(x_pos - 0.2, original_metrics, 0.4, label='Original', alpha=0.8)
            bars2 = axes[1, 0].bar(x_pos + 0.2, synthetic_metrics, 0.4, label='Synthetic', alpha=0.8)
            
            # 在柱状图上添加数值标签
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                if i == 0:  # Records用K单位
                    axes[1, 0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
                                   f'{original_metrics[i]:.1f}K', ha='center', va='bottom', fontsize=8)
                    axes[1, 0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
                                   f'{synthetic_metrics[i]:.1f}K', ha='center', va='bottom', fontsize=8)
                else:  # 其他用百分比
                    axes[1, 0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1, 
                                   f'{original_metrics[i]:.0f}%', ha='center', va='bottom', fontsize=8)
                    axes[1, 0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1, 
                                   f'{synthetic_metrics[i]:.0f}%', ha='center', va='bottom', fontsize=8)
            
            axes[1, 0].set_title('Data Quality Comparison')
            axes[1, 0].set_xlabel('Quality Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].legend()
            
            # 5. 相似性分数雷达图 - 多模型平均
            if hasattr(self, 'evaluation_results') and self.evaluation_results:
                similarity_scores = []
                labels = []
                
                # 计算指标覆盖率平均相似性
                if 'indicator_analysis' in self.evaluation_results and 'models' in self.evaluation_results['indicator_analysis']:
                    models = self.evaluation_results['indicator_analysis']['models']
                    coverage_scores = [model_data.get('coverage_similarity', 0) for model_data in models.values()]
                    if coverage_scores:
                        avg_coverage = sum(coverage_scores) / len(coverage_scores) * 100
                        similarity_scores.append(avg_coverage)
                        labels.append('Indicator\nCoverage')
                
                # 计算数值分布平均相似性
                if 'value_analysis' in self.evaluation_results and 'models' in self.evaluation_results['value_analysis']:
                    models = self.evaluation_results['value_analysis']['models']
                    dist_scores = [model_data.get('distribution_similarity', 0) for model_data in models.values() if 'distribution_similarity' in model_data]
                    if dist_scores:
                        avg_dist = sum(dist_scores) / len(dist_scores) * 100
                        similarity_scores.append(avg_dist)
                        labels.append('Value\nDistribution')
                
                # 计算时间模式平均相似性
                if 'time_analysis' in self.evaluation_results and 'models' in self.evaluation_results['time_analysis']:
                    models = self.evaluation_results['time_analysis']['models']
                    time_scores = [model_data.get('peak_hours_similarity', 0) for model_data in models.values()]
                    if time_scores:
                        avg_time = sum(time_scores) / len(time_scores) * 100
                        similarity_scores.append(avg_time)
                        labels.append('Time\nPattern')
                
                if similarity_scores:
                    axes[1, 1].bar(labels, similarity_scores, color=['skyblue', 'lightgreen', 'lightcoral'][:len(labels)])
                    axes[1, 1].set_title('Average Similarity Scores (%)')
                    axes[1, 1].set_ylabel('Similarity (%)')
                    axes[1, 1].set_ylim(0, 100)
                    
                    # 添加数值标签
                    for i, v in enumerate(similarity_scores):
                        axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
            
            # 6. 预测时间范围
            if 'predicted_date' in first_synthetic_data.columns: # 使用第一个模型的合成数据
                dates = pd.to_datetime(first_synthetic_data['predicted_date']) # 使用第一个模型的合成数据
                daily_counts = dates.dt.date.value_counts().sort_index()
                
                axes[1, 2].plot(daily_counts.index, daily_counts.values, 'o-', linewidth=2)
                axes[1, 2].set_title('Predicted Daily Activity')
                axes[1, 2].set_xlabel('Prediction Date')
                axes[1, 2].set_ylabel('Activity Count')
                axes[1, 2].tick_params(axis='x', rotation=45)
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = f'output/graph/user_{self.user_id}_comprehensive_analysis.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 分析图表已保存: {chart_file}")
            
            # 生成前16个指标的详细分析图表
            top16_chart = self._generate_top16_indicators_chart()
            
            # 生成多模型对比柱状图
            multi_model_charts = self._generate_multi_model_comparison_charts()
            
            charts = [chart_file, top16_chart] + multi_model_charts
            return charts
            
        except Exception as e:
            print(f"⚠️ 图表生成警告: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_report(self):
        """生成完整的分析报告"""
        print(f"\n📝 生成综合分析报告...")
        
        if not self.synthetic_datasets:
            print("⚠️ 没有合成数据可用于生成报告")
            return None
        
        first_model = list(self.synthetic_datasets.keys())[0]
        first_synthetic_data = self.synthetic_datasets[first_model]
        
        report_file = f'output/comprehensive_reports/user_{self.user_id}_analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # 报告标题
            f.write(f"# 用户 {self.user_id} 综合数据分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 执行摘要
            f.write("## 📋 执行摘要\n\n")
            f.write(f"本报告对用户 {self.user_id} 的健康数据进行了全面分析，包括原始数据统计、合成数据生成、")
            f.write(f"质量评估和对比分析。使用了 {len(self.synthetic_datasets)} 个不同的SDV模型")
            f.write(f"({', '.join(self.synthetic_datasets.keys())})生成合成数据，")
            f.write(f"并对数据的分布特征、时间模式和指标覆盖率进行了深入的多模型对比分析。\n\n")
            
            # 数据概览
            f.write("## 📊 数据概览\n\n")
            f.write("### 多模型数据统计\n\n")
            f.write("| 模型 | 记录数 | 指标种类 | 数值种类 | 记录比率 | 指标比率 |\n")
            f.write("|------|--------|----------|----------|----------|----------|\n")
            
            basic_stats = self.evaluation_results.get('basic_stats', {})
            orig_records = basic_stats.get('original_records', 0)
            orig_indicators = basic_stats.get('original_indicators', 0)
            value_orig = self.user_data['value'].nunique() if 'value' in self.user_data.columns else 0
            
            # 添加原始数据行
            f.write(f"| **原始数据** | {orig_records:,} | {orig_indicators} | {value_orig} | - | - |\n")
            
            # 为每个模型添加统计
            for model_name in self.synthetic_datasets.keys():
                synt_records = basic_stats.get(f'{model_name}_records', 0)
                synt_indicators = basic_stats.get(f'{model_name}_indicators', 0)
                value_synt = self.synthetic_datasets[model_name]['value'].nunique() if 'value' in self.synthetic_datasets[model_name].columns else 0
                
                record_ratio = synt_records/orig_records if orig_records > 0 else 0
                indicator_ratio = synt_indicators/orig_indicators if orig_indicators > 0 else 0
                
                f.write(f"| {model_name.upper()} | {synt_records:,} | {synt_indicators} | {value_synt} | {record_ratio:.2f}x | {indicator_ratio:.2f}x |\n")
            
            f.write("\n")
            
            # 指标分析
            if 'indicator_analysis' in self.evaluation_results:
                f.write("## 🎯 指标分析\n\n")
                indicator_analysis = self.evaluation_results['indicator_analysis']
                
                f.write("### 原始数据前5个指标\n")
                for indicator, count in indicator_analysis['original_top5'].items():
                    percentage = count / orig_records * 100
                    f.write(f"- **{indicator}**: {count} 次 ({percentage:.1f}%)\n")
                f.write("\n")
                
                # 显示所有模型的合成数据指标对比
                if 'models' in indicator_analysis and indicator_analysis['models']:
                    f.write("### 各模型指标覆盖率对比\n\n")
                    f.write("| 模型 | 前3个指标 | 覆盖率相似性 |\n")
                    f.write("|------|-----------|---------------|\n")
                    
                    for model_name, model_analysis in indicator_analysis['models'].items():
                        # 获取前3个指标
                        top3_indicators = list(model_analysis['synthetic_top5'].items())[:3]
                        top3_str = ", ".join([f"{ind}({count})" for ind, count in top3_indicators])
                        
                        coverage_sim = model_analysis['coverage_similarity'] * 100
                        f.write(f"| {model_name.upper()} | {top3_str} | {coverage_sim:.1f}% |\n")
                    
                    f.write("\n")
                    
                    # 添加前16个指标的多模型详细分析
                    f.write("### 前16个最常见指标多模型对比分析\n\n")
                    
                    # 获取原始数据的前16个指标
                    orig_indicators = self.user_data['indicator'].value_counts().head(16)
                    
                    for i, (indicator, orig_count) in enumerate(orig_indicators.items(), 1):
                        f.write(f"#### {i}. {indicator} (原始: {orig_count} 次)\n\n")
                        f.write("| 模型 | 合成计数 | 计数相似性 | 数据类型 | 分布相似性 |\n")
                        f.write("|------|----------|------------|----------|------------|\n")
                        
                        for model_name, model_analysis in indicator_analysis['models'].items():
                            if 'top16_detailed_analysis' in model_analysis:
                                top16_data = model_analysis['top16_detailed_analysis']
                                if indicator in top16_data:
                                    info = top16_data[indicator]
                                    synt_count = info.get('synthetic_count', 0)
                                    count_sim = info.get('count_similarity', 0) * 100
                                    
                                    value_analysis = info.get('value_analysis', {})
                                    data_type = value_analysis.get('type', 'unknown')
                                    
                                    if data_type == 'numerical':
                                        dist_sim = value_analysis.get('distribution_similarity', 0) * 100
                                        dist_sim_str = f"{dist_sim:.1f}%"
                                    elif data_type == 'categorical':
                                        dist_sim = value_analysis.get('value_similarity', 0) * 100
                                        dist_sim_str = f"{dist_sim:.1f}%"
                                    else:
                                        dist_sim_str = "N/A"
                                    
                                    f.write(f"| {model_name.upper()} | {synt_count} | {count_sim:.1f}% | {data_type} | {dist_sim_str} |\n")
                                else:
                                    f.write(f"| {model_name.upper()} | 0 | 0.0% | N/A | N/A |\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
            
            # 数值分析
            if 'value_analysis' in self.evaluation_results:
                f.write("## 📈 数值分析\n\n")
                value_analysis = self.evaluation_results['value_analysis']
                
                if 'original_stats' in value_analysis and 'models' in value_analysis:
                    f.write("### 多模型数值统计对比\n\n")
                    
                    orig_stats = value_analysis['original_stats']
                    
                    # 为每个统计量创建对比表
                    for stat in ['mean', 'std', 'min', 'max']:
                        if stat in orig_stats:
                            f.write(f"#### {stat.title()}统计量对比\n\n")
                            f.write("| 模型 | 数值 | 与原始数据差异(%) |\n")
                            f.write("|------|------|------------------|\n")
                            
                            orig_val = orig_stats[stat]
                            f.write(f"| **原始数据** | {orig_val:.2f} | - |\n")
                            
                            for model_name, model_value_analysis in value_analysis['models'].items():
                                if 'synthetic_stats' in model_value_analysis:
                                    synt_stats = model_value_analysis['synthetic_stats']
                                    if stat in synt_stats:
                                        synt_val = synt_stats[stat]
                                        diff_pct = abs(orig_val - synt_val) / max(abs(orig_val), 1) * 100
                                        f.write(f"| {model_name.upper()} | {synt_val:.2f} | {diff_pct:.1f}% |\n")
                            f.write("\n")
                    
                    # 分布相似性对比
                    f.write("#### 分布相似性对比\n\n")
                    f.write("| 模型 | 分布相似性 | 质量等级 |\n")
                    f.write("|------|------------|----------|\n")
                    
                    for model_name, model_value_analysis in value_analysis['models'].items():
                        if 'distribution_similarity' in model_value_analysis:
                            dist_sim = model_value_analysis['distribution_similarity'] * 100
                            
                            if dist_sim >= 90:
                                quality = "🏆 卓越"
                            elif dist_sim >= 80:
                                quality = "🌟 优秀" 
                            elif dist_sim >= 70:
                                quality = "✅ 良好"
                            elif dist_sim >= 60:
                                quality = "⚠️ 一般"
                            else:
                                quality = "❌ 需改进"
                            
                            f.write(f"| {model_name.upper()} | {dist_sim:.1f}% | {quality} |\n")
                    f.write("\n")
            
            # 时间模式分析
            if 'time_analysis' in self.evaluation_results:
                f.write("## ⏰ 时间模式分析\n\n")
                time_analysis = self.evaluation_results['time_analysis']
                
                if 'models' in time_analysis and 'original_hour_dist' in time_analysis:
                    orig_hours = time_analysis['original_hour_dist']
                    orig_peak = max(orig_hours, key=orig_hours.get)
                    
                    f.write(f"**原始数据活跃峰值**: {orig_peak}:00 ({orig_hours[orig_peak]} 次活动)\n\n")
                    
                    f.write("### 各模型时间模式对比\n\n")
                    f.write("| 模型 | 活跃峰值时间 | 峰值活动数 | 峰值时间相似性 |\n")
                    f.write("|------|--------------|------------|----------------|\n")
                    
                    for model_name, model_time_analysis in time_analysis['models'].items():
                        if 'synthetic_hour_dist' in model_time_analysis:
                            synt_hours = model_time_analysis['synthetic_hour_dist']
                            synt_peak = max(synt_hours, key=synt_hours.get)
                            peak_sim = model_time_analysis.get('peak_hours_similarity', 0) * 100
                            
                            f.write(f"| {model_name.upper()} | {synt_peak}:00 | {synt_hours[synt_peak]} | {peak_sim:.1f}% |\n")
                    f.write("\n")
            
            # 数据样例
            f.write("## 📋 数据样例\n\n")
            f.write("### 原始数据样例\n")
            f.write("```\n")
            f.write(self.user_data.head(5).to_string(index=False))
            f.write("\n```\n\n")
            
            f.write("### 合成数据样例\n")
            f.write("```\n")
            display_cols = ['indicator', 'value', 'predicted_date'] if 'predicted_date' in self.synthetic_datasets[first_model].columns else self.synthetic_datasets[first_model].columns[:5] # 使用第一个模型的合成数据
            f.write(self.synthetic_datasets[first_model][display_cols].head(5).to_string(index=False))
            f.write("\n```\n\n")
            
            # 质量评估（如果有的话）
            f.write("## 🎯 质量评估总结\n\n")
            f.write("### 主要发现\n")
            
            # 基于分析结果生成评估结论
            if 'indicator_analysis' in self.evaluation_results:
                indicator_analysis = self.evaluation_results['indicator_analysis']
                if 'models' in indicator_analysis and first_model in indicator_analysis['models']:
                    first_model_analysis = indicator_analysis['models'][first_model]
                    coverage_sim = first_model_analysis.get('coverage_similarity', 0) * 100
                    
                    if coverage_sim > 80:
                        f.write(f"✅ **指标覆盖率优秀** ({coverage_sim:.1f}%) - 合成数据很好地保持了原始数据的指标多样性\n")
                    elif coverage_sim > 60:
                        f.write(f"⚠️ **指标覆盖率良好** ({coverage_sim:.1f}%) - 大部分原始指标在合成数据中得到体现\n")
                    else:
                        f.write(f"❌ **指标覆盖率较低** ({coverage_sim:.1f}%) - 合成数据可能遗漏了一些重要指标\n")
            
            if 'value_analysis' in self.evaluation_results:
                value_analysis = self.evaluation_results['value_analysis']
                if 'models' in value_analysis and first_model in value_analysis['models']:
                    first_model_value_analysis = value_analysis['models'][first_model]
                    if 'distribution_similarity' in first_model_value_analysis:
                        dist_sim = first_model_value_analysis['distribution_similarity'] * 100
                        if dist_sim > 80:
                            f.write(f"✅ **数值分布保真度高** ({dist_sim:.1f}%) - 合成数据的统计特征与原始数据高度一致\n")
                        elif dist_sim > 60:
                            f.write(f"⚠️ **数值分布基本保持** ({dist_sim:.1f}%) - 主要统计特征得到较好保留\n")
                        else:
                            f.write(f"❌ **数值分布差异较大** ({dist_sim:.1f}%) - 需要调整模型参数以提高保真度\n")
            
            if 'time_analysis' in self.evaluation_results:
                time_analysis = self.evaluation_results['time_analysis']
                if 'models' in time_analysis and first_model in time_analysis['models']:
                    first_model_time_analysis = time_analysis['models'][first_model]
                    time_sim = first_model_time_analysis.get('peak_hours_similarity', 0) * 100
                    if time_sim > 80:
                        f.write(f"✅ **时间模式高度一致** ({time_sim:.1f}%) - 活动峰值时间模式得到很好保留\n")
                    elif time_sim > 60:
                        f.write(f"⚠️ **时间模式基本一致** ({time_sim:.1f}%) - 整体活动规律相似\n")
                    else:
                        f.write(f"❌ **时间模式差异明显** ({time_sim:.1f}%) - 活动时间分布有显著变化\n")
            
            f.write("\n")
            f.write("### 建议\n")
            f.write("- 根据相似性分数调整模型参数以提高数据质量\n")
            f.write("- 关注指标覆盖率，确保重要健康指标不被遗漏\n")
            f.write("- 验证时间模式的合理性，保持用户的活动规律\n")
            f.write("- 定期评估合成数据质量，持续优化生成效果\n\n")
            
            # 技术信息
            f.write("## 🔧 技术信息\n\n")
            f.write(f"- **模型类型**: PAR (Probabilistic AutoRegressive)\n")
            f.write(f"- **训练数据量**: {orig_records:,} 条记录\n")
            f.write(f"- **生成数据量**: {synt_records:,} 条记录\n")
            f.write(f"- **数据时间范围**: {(self.last_date - timedelta(days=30)).date()} 到 {self.last_date.date()}\n")
            f.write(f"- **报告生成工具**: ComprehensiveEvaluator v1.0\n\n")
            
            f.write("---\n")
            f.write("*此报告由SDV-Theta综合评测器自动生成*")
        
        print(f"✅ 分析报告已保存: {report_file}")
        
        # 同时保存数据文件
        data_file = f'output/comprehensive_reports/user_{self.user_id}_synthetic_data.csv'
        self.synthetic_datasets[first_model].to_csv(data_file, index=False) # 使用第一个模型的合成数据
        print(f"✅ 合成数据已保存: {data_file}")
        
        return report_file

def main():
    """主函数"""
    print("🚀 SDV-Theta 综合数据评测器 - 分析和报告生成")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("📖 使用方法:")
        print(f"  python3 {sys.argv[0]} <用户ID> [模型列表] [最大记录数]")
        print(f"\n💡 示例:")
        print(f"  python3 {sys.argv[0]} 169                    # 用户169，分析所有可用模型")
        print(f"  python3 {sys.argv[0]} 169 ctgan,par         # 用户169，只分析CTGAN和PAR模型")
        print(f"  python3 {sys.argv[0]} 169 all 5000          # 用户169，所有模型，最多5000条记录")
        print(f"\n📋 支持的模型: gaussian_copula, ctgan, copulagan, tvae, par")
        print(f"\n⚠️  注意: 请先运行 python3 run_all_models.py 生成合成数据")
        
        # 显示可用用户
        evaluator = ComprehensiveEvaluator()
        evaluator.load_data()
        evaluator.analyze_users()
        return
    
    # 解析参数
    try:
        user_id = int(sys.argv[1])
        models_arg = sys.argv[2] if len(sys.argv) > 2 else 'all'
        max_records = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        # 解析模型列表
        if models_arg.lower() == 'all':
            models = None  # 使用所有支持的模型
        else:
            models = [m.strip() for m in models_arg.split(',')]
            # 验证模型名称
            valid_models = ['gaussian_copula', 'ctgan', 'copulagan', 'tvae', 'par']
            invalid_models = [m for m in models if m not in valid_models]
            if invalid_models:
                print(f"❌ 无效的模型名称: {invalid_models}")
                print(f"📋 支持的模型: {valid_models}")
                return
        
        if max_records and max_records < 100:
            print("❌ 最大记录数必须大于100")
            return
            
    except ValueError:
        print("❌ 用户ID必须是数字")
        return
    
    # 执行综合评测
    evaluator = ComprehensiveEvaluator()
    
    # 1. 数据加载和用户选择
    if not evaluator.load_data():
        return
    
    if not evaluator.select_user(user_id, max_records=max_records):
        return
    
    # 2. 加载合成数据
    loaded_models = evaluator.load_synthetic_data(models=models)
    if not loaded_models:
        return
    
    # 3. 统计分析
    evaluator.statistical_analysis()
    
    # 4. 生成可视化
    chart_files = evaluator.generate_visualizations()
    
    # 5. 生成报告
    report_file = evaluator.generate_report()
    
    # 6. 总结
    print(f"\n🎉 用户 {user_id} 综合评测完成！")
    print(f"📊 分析的模型: {', '.join(loaded_models)}")
    if isinstance(chart_files, list) and chart_files:
        print(f"\n📊 生成的图表文件:")
        for i, chart_file in enumerate(chart_files, 1):
            if chart_file:
                chart_name = chart_file.split('/')[-1]
                if 'comprehensive_analysis' in chart_name:
                    print(f"  {i}. 综合分析图表: {chart_file}")
                elif 'top16_indicators' in chart_name:
                    print(f"  {i}. 前16个指标详细分析: {chart_file}")
                elif 'coverage_comparison' in chart_name:
                    print(f"  {i}. 指标覆盖率对比图: {chart_file}")
                elif 'top10_indicators_comparison' in chart_name:
                    print(f"  {i}. 前10个指标分布对比图: {chart_file}")
                elif 'statistics_comparison' in chart_name:
                    print(f"  {i}. 数值统计对比图: {chart_file}")
                elif 'time_pattern_comparison' in chart_name:
                    print(f"  {i}. 时间模式对比图: {chart_file}")
                else:
                    print(f"  {i}. 其他图表: {chart_file}")
    elif chart_files:
        print(f"📊 分析图表: {chart_files}")
    
    print(f"📄 分析报告: {report_file}")
    print(f"📁 输出目录: output/comprehensive_reports/")
    print(f"📁 图表目录: output/graph/")
    print(f"\n💡 如需重新生成合成数据，请运行: python3 run_all_models.py")

if __name__ == "__main__":
    main() 