#!/usr/bin/env python3
"""
PAR 模型 - 基于概率自回归的序列数据合成
适用场景: 时间序列数据，具有序列依赖关系的数据
增强模式: 保持JSON/文本结构的复杂医疗数据处理
"""

import sys
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import *

class SequentialTextProcessor:
    """序列文本处理器，保持时间序列中的文本结构"""
    
    def __init__(self):
        self.medical_terms = [
            '心律', '心电图', '超声', '血压', '血糖', '甲状腺', '结节', 
            '窦性', '心动', '正常', '异常', '检查', '报告', '分析'
        ]
    
    def create_sequence_features(self, df, value_col='value'):
        """为序列数据创建结构化特征"""
        print("🏗️ 创建序列结构化特征...")
        
        df_enhanced = df.copy()
        
        # 文本分类特征
        df_enhanced['text_category'] = df[value_col].apply(self._classify_medical_text)
        df_enhanced['medical_term_count'] = df[value_col].apply(self._count_medical_terms)
        df_enhanced['text_complexity'] = df[value_col].apply(self._assess_complexity)
        df_enhanced['report_type'] = df[value_col].apply(self._classify_report_type)
        df_enhanced['has_measurements'] = df[value_col].apply(self._detect_measurements)
        
        # 序列位置特征
        df_enhanced['sequence_position'] = df_enhanced.groupby('user_id').cumcount()
        
        print(f"✓ 序列特征创建完成")
        return df_enhanced
    
    def _classify_medical_text(self, text):
        if pd.isna(text):
            return 'empty'
        text_str = str(text)
        
        if re.search(r'\d+\.\s*[^\n]+.*\n.*\d+\.', text_str):
            return 'detailed_report'
        elif any(term in text_str for term in ['心电图', '超声', '血压']):
            return 'diagnostic_test'
        elif '正常' in text_str:
            return 'normal_result'
        elif '异常' in text_str:
            return 'abnormal_result'
        elif re.search(r'\d+(\.\d+)?\s*(mmol|mg|%)', text_str):
            return 'measurement'
        else:
            return 'general_text'
    
    def _count_medical_terms(self, text):
        if pd.isna(text):
            return 0
        text_str = str(text)
        return sum(1 for term in self.medical_terms if term in text_str)
    
    def _assess_complexity(self, text):
        if pd.isna(text):
            return 'simple'
        text_str = str(text)
        
        if len(text_str) > 50 and '\n' in text_str:
            return 'complex'
        elif len(text_str) > 20:
            return 'medium'
        else:
            return 'simple'
    
    def _classify_report_type(self, text):
        if pd.isna(text):
            return 'none'
        text_str = str(text)
        
        if '心电图' in text_str:
            return 'ecg_report'
        elif '超声' in text_str:
            return 'ultrasound_report'
        elif '血' in text_str:
            return 'blood_test'
        elif re.search(r'\d+\.\s*', text_str):
            return 'structured_report'
        else:
            return 'general_report'
    
    def _detect_measurements(self, text):
        if pd.isna(text):
            return 0
        text_str = str(text)
        
        if re.search(r'\d+(\.\d+)?\s*(mmol|mg|%|次|分钟)', text_str):
            return 1
        else:
            return 0

def preprocess_enhanced(df, sample_size=3000, enhanced_mode=False):
    """增强预处理，可选择是否保持文本结构"""
    if enhanced_mode:
        print("🔧 使用增强模式预处理...")
        
        # 智能采样保持序列完整性
        if sample_size and len(df) > sample_size:
            users = df['user_id'].unique()
            sample_users = np.random.choice(users, size=min(sample_size//20, len(users)), replace=False)
            df = df[df['user_id'].isin(sample_users)]
            print(f"✓ 按用户采样，保留 {len(sample_users)} 个用户的完整序列")
        
        # 删除不必要的列
        columns_to_drop = ['id', 'source_table_id', 'comment', 'indicator_id', 'deleted']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 使用序列文本处理器
        text_processor = SequentialTextProcessor()
        df_enhanced = text_processor.create_sequence_features(df)
        
        # 保留更多医疗指标
        if 'indicator' in df_enhanced.columns:
            top_indicators = df_enhanced['indicator'].value_counts().head(40).index
            df_enhanced.loc[~df_enhanced['indicator'].isin(top_indicators), 'indicator'] = 'other_medical_indicator'
            print(f"✓ 保留前40个重要医疗指标")
        
        # 简化时间处理
        if 'create_time' in df_enhanced.columns:
            try:
                time_data = pd.to_datetime(df_enhanced['create_time'])
                df_enhanced['create_time_hour'] = time_data.dt.hour
                df_enhanced = df_enhanced.drop(columns=['create_time'])
                print(f"✓ create_time 转换为小时特征")
            except:
                df_enhanced = df_enhanced.drop(columns=['create_time'])
        
        # 删除其他时间列
        time_columns = ['start_time', 'end_time', 'update_time']
        for col in time_columns:
            if col in df_enhanced.columns:
                df_enhanced = df_enhanced.drop(columns=[col])
        
        # 保存value列样本用于重构
        if 'value' in df_enhanced.columns:
            df_enhanced['value_simplified'] = df_enhanced['value'].apply(
                lambda x: str(x)[:50] if pd.notna(x) else 'empty'
            )
            print("✓ 保存value列样本用于重构")
        
        return df_enhanced, text_processor
    else:
        # 使用原始预处理
        return preprocess_data(df, sample_size=sample_size, reduce_cardinality=True), None

def prepare_sequential_data(df, enhanced_mode=False):
    """为PAR模型准备序列数据"""
    print("正在为PAR模型准备序列数据...")
    
    # 确保有用户ID列用于分组序列
    if 'user_id' not in df.columns:
        print("❌ 数据中没有user_id列，PAR模型需要序列标识符")
        return None
    
    # 排序数据以确保时间顺序
    if enhanced_mode:
        df = df.sort_values(['user_id', 'sequence_position'])
    else:
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
    
    # 过滤掉序列太短的用户
    min_length = 3 if enhanced_mode else 2
    valid_users = sequence_lengths[sequence_lengths >= min_length].index
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    if len(df_filtered) < len(df):
        removed_count = len(df) - len(df_filtered)
        print(f"✓ 已移除 {removed_count} 条记录（来自序列长度<{min_length}的用户）")
    
    return df_filtered

def reconstruct_text_structure(synthetic_df, text_processor=None, enhanced_mode=False):
    """重构文本结构"""
    if not enhanced_mode or text_processor is None:
        return synthetic_df
    
    print("🔄 重构医疗文本结构...")
    synthetic_df = synthetic_df.copy()
    
    def reconstruct_medical_value(row):
        text_category = row.get('text_category', 'general_text')
        report_type = row.get('report_type', 'general_report')
        complexity = row.get('text_complexity', 'simple')
        medical_count = row.get('medical_term_count', 0)
        has_measurements = row.get('has_measurements', 0)
        
        if report_type == 'ecg_report':
            if complexity == 'complex':
                return "1.窦性心律\n2.心率正常\n3.未见明显异常"
            else:
                return "窦性心律，心率正常"
        elif report_type == 'ultrasound_report':
            if text_category == 'abnormal_result':
                return "超声检查：发现结节，建议进一步检查"
            else:
                return "超声检查：各脏器大小形态正常"
        elif report_type == 'blood_test':
            if has_measurements:
                return "血糖:5.2mmol/L，总胆固醇:4.1mmol/L"
            else:
                return "血液检查各项指标正常"
        elif report_type == 'structured_report':
            if medical_count > 2:
                return "1.检查完成\n2.各项指标在正常范围\n3.建议定期复查"
            else:
                return "检查结果正常"
        elif text_category == 'normal_result':
            return "正常"
        elif text_category == 'abnormal_result':
            return "异常，需要关注"
        elif text_category == 'measurement':
            return "测量值在正常范围内"
        else:
            return "检查完成"
    
    # 重构value列
    synthetic_df['value'] = synthetic_df.apply(reconstruct_medical_value, axis=1)
    
    # 删除临时特征列
    feature_cols = ['text_category', 'medical_term_count', 'text_complexity', 
                   'report_type', 'has_measurements', 'value_simplified',
                   'sequence_position']
    for col in feature_cols:
        if col in synthetic_df.columns:
            synthetic_df = synthetic_df.drop(columns=[col])
    
    print("✅ 文本结构重构完成")
    return synthetic_df

def create_sequential_metadata(df):
    """创建序列数据的元数据"""
    print("正在创建序列数据元数据...")
    
    # 添加序列索引
    df['sequence_index'] = df.groupby('user_id').cumcount()
    
    # 使用SingleTableMetadata替代Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # 设置user_id为序列键
    metadata.set_sequence_key('user_id')
    
    # 设置序列索引
    if 'sequence_index' in df.columns:
        metadata.update_column('sequence_index', sdtype='numerical')
        metadata.set_sequence_index('sequence_index')
        print(f"✓ 设置sequence_index为序列索引")
    else:
        # 如果有时间列，设置为序列索引
        time_cols = ['start_time', 'end_time', 'create_time', 'update_time']
        for col in time_cols:
            if col in df.columns:
                try:
                    metadata.set_sequence_index(col)
                    print(f"✓ 已设置 {col} 为序列索引")
                    break
                except:
                    continue
    
    print("✓ 序列元数据创建完成")
    return metadata

def main():
    import argparse
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='PAR序列数据合成模型')
    parser.add_argument('--enhanced', action='store_true', 
                       help='使用增强模式保持JSON/文本结构')
    parser.add_argument('--sample-size', type=int, default=3000,
                       help='采样大小 (默认: 3000)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数 (默认: 5)')
    parser.add_argument('--sequences', type=int, default=50,
                       help='生成序列数量 (默认: 50)')
    
    args = parser.parse_args()
    
    mode_desc = "增强模式 - 保持JSON/文本结构" if args.enhanced else "标准模式"
    print_model_info("PAR", f"基于概率自回归的序列数据合成模型 ({mode_desc})")
    
    # 数据路径
    data_path = "source_data/th_series_data.csv"
    output_dir = "output/par_enhanced" if args.enhanced else "output/par"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 预处理数据
        if args.enhanced:
            # 先预采样以优化内存
            df = df.sample(n=min(5000, len(df)), random_state=42)
            print(f"✓ 预采样到 {len(df)} 条记录以优化内存使用")
            
        df_processed, text_processor = preprocess_enhanced(
            df, sample_size=args.sample_size, enhanced_mode=args.enhanced
        )
        
        # 3. 准备序列数据
        df_sequential = prepare_sequential_data(df_processed, enhanced_mode=args.enhanced)
        if df_sequential is None:
            print("❌ 无法准备序列数据，退出PAR模型训练")
            return
        
        # 删除原始value列（如果使用增强模式）
        if args.enhanced and 'value' in df_sequential.columns:
            df_sequential = df_sequential.drop(columns=['value'])
        
        # 4. 创建序列元数据
        metadata = create_sequential_metadata(df_sequential)
        
        # 5. 创建 PAR 合成器
        print("创建 PAR 合成器...")
        synthesizer = PARSynthesizer(
            metadata=metadata,
            epochs=args.epochs,
            context_columns=None,
            verbose=True
        )
        
        # 6. 训练并生成数据
        print("正在训练PAR模型...")
        start_time = datetime.now()
        
        synthesizer.fit(df_sequential)
        
        train_time = datetime.now() - start_time
        print(f"✓ PAR模型训练完成，耗时: {train_time}")
        
        # 7. 生成合成数据
        print("正在生成合成序列数据...")
        num_sequences = min(args.sequences, len(df_sequential['user_id'].unique()))
        if args.enhanced:
            num_sequences = min(10, num_sequences)  # 增强模式使用较少序列
            
        synthetic_data = synthesizer.sample(num_sequences=num_sequences)
        
        print(f"✓ 成功生成 {len(synthetic_data)} 行合成序列数据")
        
        # 8. 重构文本结构（如果使用增强模式）
        synthetic_data_final = reconstruct_text_structure(
            synthetic_data, text_processor, enhanced_mode=args.enhanced
        )
        
        # 9. 评估模型（增强模式跳过评估以节省资源）
        quality_report = None
        if not args.enhanced:
            print("正在评估PAR模型...")
            try:
                quality_report = evaluate_model(
                    df_sequential, 
                    synthetic_data_final, 
                    metadata, 
                    "Enhanced_PAR" if args.enhanced else "PAR"
                )
            except Exception as e:
                print(f"⚠️ 序列数据评估遇到问题: {e}")
        
        # 10. 保存结果
        save_results(
            synthetic_data_final, 
            "Enhanced_PAR" if args.enhanced else "PAR", 
            output_dir, 
            metadata, 
            quality_report, 
            train_time
        )
        
        # 11. 打印详细统计
        print("\n=== 序列数据统计对比 ===")
        print(f"\n原始数据:")
        print(f"- 总记录数: {len(df_sequential)}")
        print(f"- 用户数量: {df_sequential['user_id'].nunique()}")
        print(f"- 平均每用户记录数: {len(df_sequential) / df_sequential['user_id'].nunique():.1f}")
        
        print(f"\n合成数据:")
        print(f"- 总记录数: {len(synthetic_data_final)}")
        print(f"- 用户数量: {synthetic_data_final['user_id'].nunique()}")
        print(f"- 平均每用户记录数: {len(synthetic_data_final) / synthetic_data_final['user_id'].nunique():.1f}")
        
        if args.enhanced:
            print("\n📊 增强模式结构对比:")
            print("原始数据样例:")
            original_sample = df[['user_id', 'indicator', 'value']].head(3)
            print(original_sample.to_string(index=False))
            
            print("\n增强PAR生成数据样例:")
            enhanced_sample = synthetic_data_final[['user_id', 'indicator', 'value']].head(3)
            print(enhanced_sample.to_string(index=False))
        
        print(f"\n✅ PAR 模型执行完成！")
        print(f"结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 