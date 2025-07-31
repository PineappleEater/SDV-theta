#!/usr/bin/env python3
"""
🎯 模型优化效果验证脚本
用于对比优化前后的模型性能，验证参数调优效果
"""

import pandas as pd
import time
import os

def analyze_optimization_results():
    """分析优化效果"""
    print("🎯 SDV-Theta 模型优化效果验证")
    print("=" * 60)
    
    # 1. 检查是否有新的训练结果
    models = ['gaussian_copula', 'ctgan', 'copulagan', 'tvae', 'par']
    
    print("📊 检查模型训练结果...")
    results = {}
    
    for model in models:
        output_dir = f"output/{model}"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if csv_files:
                # 获取最新的CSV文件
                latest_csv = max([os.path.join(output_dir, f) for f in csv_files], 
                                key=os.path.getmtime)
                modification_time = os.path.getmtime(latest_csv)
                
                # 检查文件是否是最近生成的（1小时内）
                if time.time() - modification_time < 3600:
                    status = "✅ 新生成"
                    try:
                        df = pd.read_csv(latest_csv)
                        record_count = len(df)
                        indicator_count = df['indicator'].nunique() if 'indicator' in df.columns else 0
                    except:
                        record_count = 0
                        indicator_count = 0
                else:
                    status = "🕐 旧文件"
                    record_count = 0
                    indicator_count = 0
            else:
                status = "❌ 无结果"
                record_count = 0
                indicator_count = 0
        else:
            status = "❌ 目录不存在"
            record_count = 0
            indicator_count = 0
            
        results[model] = {
            'status': status,
            'records': record_count,
            'indicators': indicator_count
        }
        
        print(f"  {model.upper():15s}: {status:10s} ({record_count} 条记录, {indicator_count} 指标)")
    
    print()
    
    # 2. 如果有新结果，进行快速分析
    new_results = [model for model, data in results.items() 
                   if data['status'] == "✅ 新生成" and data['records'] > 0]
    
    if new_results:
        print("🎉 发现新的训练结果！")
        print(f"✅ 成功训练的模型: {', '.join([m.upper() for m in new_results])}")
        
        # 3. 对比分析（简化版）
        print("\n📈 快速性能分析...")
        
        # 加载原始数据作为基准
        try:
            df_orig = pd.read_csv('source_data/th_series_data.csv', low_memory=False)
            user_169 = df_orig[df_orig['user_id'] == 169].sample(n=min(5000, len(df_orig[df_orig['user_id'] == 169])), random_state=42)
            orig_indicators = user_169['indicator'].value_counts().head(5)
            orig_total = len(user_169)
            
            print(f"📊 原始基准数据 (前5指标):")
            for i, (ind, count) in enumerate(orig_indicators.items(), 1):
                freq = count / orig_total * 100
                print(f"  {i}. {ind[:30]:30s}: {freq:5.1f}%")
            
            print("\n🔍 模型表现对比:")
            
            for model in new_results:
                try:
                    output_dir = f"output/{model}"
                    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                    if csv_files:
                        latest_csv = max([os.path.join(output_dir, f) for f in csv_files], 
                                        key=os.path.getmtime)
                        df_synt = pd.read_csv(latest_csv)
                        synt_indicators = df_synt['indicator'].value_counts()
                        synt_total = len(df_synt)
                        
                        print(f"\n--- {model.upper()} ---")
                        
                        # 计算前5指标的频率误差
                        errors = []
                        for indicator in orig_indicators.head(5).index:
                            orig_freq = orig_indicators[indicator] / orig_total * 100
                            synt_count = synt_indicators.get(indicator, 0)
                            synt_freq = synt_count / synt_total * 100 if synt_total > 0 else 0
                            error = abs(orig_freq - synt_freq)
                            errors.append(error)
                            print(f"  {indicator[:25]:25s}: {synt_freq:5.1f}% (目标: {orig_freq:5.1f}%, 误差: {error:5.2f}%)")
                        
                        avg_error = sum(errors) / len(errors)
                        print(f"  📊 平均频率误差: {avg_error:.2f}%")
                        
                        # 判断优化效果
                        if avg_error < 0.5:
                            print(f"  🎉 优秀表现！")
                        elif avg_error < 1.0:
                            print(f"  ✅ 良好表现")
                        elif avg_error < 1.5:
                            print(f"  ⚠️  需要改进")
                        else:
                            print(f"  ❌ 表现不佳")
                            
                except Exception as e:
                    print(f"  ❌ {model} 分析失败: {e}")
        
        except Exception as e:
            print(f"❌ 原始数据加载失败: {e}")
    
    else:
        print("⏳ 暂无新的训练结果，请等待模型训练完成...")
        print("💡 可能的原因:")
        print("  1. 模型还在训练中（TVAE预计5-8分钟）")
        print("  2. 训练过程中出现错误")
        print("  3. 路径配置问题")
    
    print("\n" + "=" * 60)
    print("💡 提示: 可以定期运行此脚本检查优化效果")
    print("📈 完整分析请运行: python3 comprehensive_evaluator.py 169")

if __name__ == "__main__":
    analyze_optimization_results() 