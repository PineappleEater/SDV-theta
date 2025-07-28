#!/usr/bin/env python3
"""
模型对比分析脚本 - 对比不同SDV模型的生成结果
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def load_synthetic_data():
    """加载所有模型的合成数据"""
    models_data = {}
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在，请先运行模型生成数据")
        return {}
    
    for model_name in ['gaussian_copula', 'ctgan', 'copulagan', 'tvae', 'par']:
        model_dir = os.path.join(output_dir, model_name)
        if os.path.exists(model_dir):
            # 查找合成数据文件
            data_files = glob.glob(os.path.join(model_dir, "*_synthetic_data.csv"))
            if data_files:
                try:
                    df = pd.read_csv(data_files[0])
                    models_data[model_name] = df
                    print(f"✅ 加载 {model_name} 数据: {df.shape}")
                except Exception as e:
                    print(f"❌ 加载 {model_name} 数据失败: {e}")
            else:
                print(f"⚠️  未找到 {model_name} 的合成数据文件")
    
    return models_data

def load_quality_scores():
    """加载所有模型的质量分数"""
    quality_scores = {}
    
    output_dir = "output"
    for model_name in ['gaussian_copula', 'ctgan', 'copulagan', 'tvae', 'par']:
        model_dir = os.path.join(output_dir, model_name)
        if os.path.exists(model_dir):
            summary_files = glob.glob(os.path.join(model_dir, "*_summary.txt"))
            if summary_files:
                try:
                    with open(summary_files[0], 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 提取质量分数
                        for line in content.split('\n'):
                            if '质量分数:' in line and '%' in line:
                                score_str = line.split('质量分数:')[1].strip()
                                if score_str != '无法获取':
                                    score = float(score_str.replace('%', ''))
                                    quality_scores[model_name] = score
                                    break
                except Exception as e:
                    print(f"❌ 读取 {model_name} 质量分数失败: {e}")
    
    return quality_scores

def compare_data_distributions(models_data):
    """对比数据分布"""
    print("\n📊 数据分布对比分析")
    print("="*60)
    
    if not models_data:
        print("❌ 没有可用的模型数据进行对比")
        return
    
    # 获取数值列
    first_model = list(models_data.keys())[0]
    numeric_columns = models_data[first_model].select_dtypes(include=[np.number]).columns
    
    comparison_results = {}
    
    for col in numeric_columns[:5]:  # 只对比前5个数值列
        print(f"\n列: {col}")
        print("-" * 40)
        
        col_comparison = {}
        for model_name, data in models_data.items():
            if col in data.columns:
                stats = data[col].describe()
                col_comparison[model_name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['50%']
                }
                print(f"{model_name:15} | 均值: {stats['mean']:.2f} | 标准差: {stats['std']:.2f}")
        
        comparison_results[col] = col_comparison
    
    return comparison_results

def analyze_sequential_patterns(models_data):
    """分析序列模式（特别针对PAR模型）"""
    print("\n🔄 序列模式分析")
    print("="*60)
    
    for model_name, data in models_data.items():
        if 'user_id' in data.columns:
            user_counts = data['user_id'].value_counts()
            print(f"\n{model_name} 序列特征:")
            print(f"  - 用户数量: {data['user_id'].nunique()}")
            print(f"  - 总记录数: {len(data)}")
            print(f"  - 平均序列长度: {user_counts.mean():.1f}")
            print(f"  - 序列长度范围: {user_counts.min()} - {user_counts.max()}")

def create_comparison_report(models_data, quality_scores):
    """创建对比报告"""
    print("\n📋 生成综合对比报告")
    
    report_content = []
    report_content.append("=" * 80)
    report_content.append("SDV 模型对比分析报告")
    report_content.append("=" * 80)
    report_content.append(f"生成时间: {datetime.now()}")
    report_content.append("")
    
    # 1. 基本信息对比
    report_content.append("1. 基本信息对比")
    report_content.append("-" * 40)
    report_content.append(f"{'模型名称':<15} | {'数据行数':<10} | {'数据列数':<10} | {'质量分数':<10}")
    report_content.append("-" * 60)
    
    for model_name, data in models_data.items():
        quality_score = quality_scores.get(model_name, 'N/A')
        score_str = f"{quality_score:.2f}%" if isinstance(quality_score, (int, float)) else str(quality_score)
        report_content.append(f"{model_name:<15} | {len(data):<10} | {len(data.columns):<10} | {score_str:<10}")
    
    report_content.append("")
    
    # 2. 质量分数排名
    if quality_scores:
        report_content.append("2. 质量分数排名")
        report_content.append("-" * 40)
        
        sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, score) in enumerate(sorted_scores, 1):
            report_content.append(f"{i}. {model_name}: {score:.2f}%")
        
        report_content.append("")
    
    # 3. 序列模式分析（如果有PAR模型）
    if 'par' in models_data:
        report_content.append("3. 序列模式分析")
        report_content.append("-" * 40)
        
        for model_name, data in models_data.items():
            if 'user_id' in data.columns:
                user_counts = data['user_id'].value_counts()
                report_content.append(f"• {model_name}:")
                report_content.append(f"  - 用户数: {data['user_id'].nunique()}, 总记录: {len(data)}")
                report_content.append(f"  - 平均序列长度: {user_counts.mean():.1f}")
        
        report_content.append("")
    
    # 4. 模型特点分析
    report_content.append("4. 模型特点分析")
    report_content.append("-" * 40)
    
    model_descriptions = {
        'gaussian_copula': '经典统计模型，训练快速，适合结构化数据',
        'ctgan': '深度学习GAN模型，生成质量高，训练时间较长',
        'copulagan': '混合模型，平衡速度和质量',
        'tvae': '变分自编码器，适合特征学习和降维',
        'par': '概率自回归模型，专门用于序列/时间序列数据'
    }
    
    for model_name in models_data.keys():
        description = model_descriptions.get(model_name, '未知模型')
        report_content.append(f"• {model_name}: {description}")
    
    report_content.append("")
    
    # 5. 推荐建议
    report_content.append("5. 使用建议")
    report_content.append("-" * 40)
    
    if quality_scores:
        best_model = max(quality_scores.items(), key=lambda x: x[1])
        report_content.append(f"• 质量最佳: {best_model[0]} (分数: {best_model[1]:.2f}%)")
    
    report_content.append("• 速度优先: gaussian_copula")
    report_content.append("• 质量优先: ctgan")
    report_content.append("• 平衡选择: copulagan")
    report_content.append("• 特征学习: tvae")
    report_content.append("• 序列/时间序列: par")
    
    # 保存报告
    report_file = "output/models_comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"✅ 对比报告已保存到: {report_file}")
    
    # 打印到控制台
    print("\n" + "\n".join(report_content))

def main():
    print("🔍 开始模型对比分析")
    
    # 1. 加载数据
    models_data = load_synthetic_data()
    if not models_data:
        print("❌ 没有找到任何模型数据，请先运行 run_all_models.py")
        return
    
    # 2. 加载质量分数
    quality_scores = load_quality_scores()
    
    # 3. 对比数据分布
    compare_data_distributions(models_data)
    
    # 4. 分析序列模式
    analyze_sequential_patterns(models_data)
    
    # 5. 创建综合报告
    create_comparison_report(models_data, quality_scores)
    
    print(f"\n🎉 模型对比分析完成！")

if __name__ == "__main__":
    main() 