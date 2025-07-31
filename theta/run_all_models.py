#!/usr/bin/env python3
"""
批量模型运行器 - 运行所有五个优化版SDV模型
✨ 功能: 智能数据处理 + 可视化进度条 + 详细对比报告
一键运行: GaussianCopula, CTGAN, CopulaGAN, TVAE, PAR
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from utils import progress_bar

def print_header():
    """打印程序头部信息"""
    print("=" * 100)
    print("🚀 SDV-Theta 批量模型运行器")
    print("🤖 将运行所有五个优化版模型:")
    print("   1. GaussianCopula - 经典统计模型 🥇")
    print("   2. CTGAN - 条件生成对抗网络")
    print("   3. CopulaGAN - 混合Copula+GAN")
    print("   4. TVAE - 表格变分自编码器 (重点优化)")
    print("   5. PAR - 概率自回归序列模型")
    print("=" * 100)

def run_model(model_script, model_name, timeout=1200):
    """运行单个模型并收集结果"""
    print(f"\n🔄 启动 {model_name} 模型...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 运行模型脚本
        result = subprocess.run(
            [sys.executable, model_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 执行成功!")
            print(f"⏱️  执行时间: {execution_time:.2f} 秒")
            
            # 提取质量分数（如果有的话）
            quality_score = extract_quality_score(result.stdout)
            
            return {
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_score,
                'output': result.stdout,
                'error': None
            }
        else:
            print(f"❌ {model_name} 执行失败!")
            print(f"错误代码: {result.returncode}")
            print(f"错误信息: {result.stderr}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'quality_score': None,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 执行超时 (>{timeout}秒)")
        return {
            'success': False,
            'execution_time': timeout,
            'quality_score': None,
            'output': None,
            'error': "Timeout"
        }
        
    except Exception as e:
        print(f"💥 {model_name} 执行异常: {e}")
        return {
            'success': False,
            'execution_time': 0,
            'quality_score': None,
            'output': None,
            'error': str(e)
        }

def extract_quality_score(output):
    """从输出中提取质量分数"""
    if not output:
        return None
        
    lines = output.split('\n')
    for line in lines:
        if '质量分数:' in line or 'quality score:' in line.lower():
            try:
                # 提取百分比数字
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    return float(match.group(1))
            except:
                pass
    return None

def generate_comparison_report(results):
    """生成模型对比报告"""
    print("\n" + "=" * 100)
    print("📊 模型对比报告")
    print("=" * 100)
    
    # 创建报告文件
    report_file = f"output/models_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs('output', exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# SDV模型对比报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 📊 执行结果概览\n\n")
        
        # 控制台输出
        print("\n📈 执行结果:")
        print("=" * 80)
        print(f"{'模型名称':<20} {'状态':<8} {'执行时间':<12} {'质量分数':<12} {'等级':<10}")
        print("-" * 80)
        
        # 详细结果表格
        f.write("| 模型名称 | 执行状态 | 执行时间(秒) | 质量分数(%) | 质量等级 | 备注 |\n")
        f.write("|----------|----------|-------------|-------------|----------|------|\n")
        
        successful_models = 0
        total_time = 0
        quality_scores = []
        
        for model_name, result in results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            exec_time = f"{result['execution_time']:.1f}s"
            
            if result['quality_score'] is not None:
                quality_str = f"{result['quality_score']:.1f}%"
                quality_scores.append(result['quality_score'])
                
                # 质量等级评估
                score = result['quality_score']
                if score >= 90:
                    quality_level = "🏆 卓越"
                elif score >= 80:
                    quality_level = "🌟 优秀"
                elif score >= 70:
                    quality_level = "✅ 良好"
                elif score >= 60:
                    quality_level = "⚠️ 一般"
                else:
                    quality_level = "❌ 需改进"
            else:
                quality_str = "N/A"
                quality_level = "未知"
            
            # 备注
            if result['success']:
                note = "正常完成"
                successful_models += 1
            elif result['error'] == "Timeout":
                note = "执行超时"
            else:
                note = "执行错误"
            
            total_time += result['execution_time']
            
            # 控制台输出
            print(f"{model_name:<20} {status:<8} {exec_time:<12} {quality_str:<12} {quality_level:<10}")
            
            # 报告文件输出
            f.write(f"| {model_name} | {status} | {result['execution_time']:.1f} | {quality_str} | {quality_level} | {note} |\n")
        
        print("-" * 80)
        print(f"成功模型: {successful_models}/5")
        print(f"总执行时间: {total_time/60:.1f} 分钟")
        
        # 统计信息
        f.write(f"\n## 📈 统计信息\n\n")
        f.write(f"- **成功模型数**: {successful_models}/5\n")
        f.write(f"- **总执行时间**: {total_time/60:.1f} 分钟\n")
        f.write(f"- **平均执行时间**: {total_time/5/60:.1f} 分钟\n")
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            best_model = max(results.items(), key=lambda x: x[1]['quality_score'] if x[1]['quality_score'] else 0)
            
            print(f"平均质量分数: {avg_quality:.1f}%")
            print(f"最佳模型: {best_model[0]} ({best_model[1]['quality_score']:.1f}%)")
            
            f.write(f"- **平均质量分数**: {avg_quality:.1f}%\n")
            f.write(f"- **最佳模型**: {best_model[0]} ({best_model[1]['quality_score']:.1f}%)\n")
            f.write(f"- **质量分数范围**: {min(quality_scores):.1f}% - {max(quality_scores):.1f}%\n")
        
        # 详细结果
        f.write(f"\n## 🔍 详细结果\n\n")
        for model_name, result in results.items():
            f.write(f"### {model_name}\n\n")
            f.write(f"- **执行状态**: {'成功' if result['success'] else '失败'}\n")
            f.write(f"- **执行时间**: {result['execution_time']:.2f} 秒\n")
            
            if result['quality_score']:
                f.write(f"- **质量分数**: {result['quality_score']:.1f}%\n")
            
            if result['error']:
                f.write(f"- **错误信息**: {result['error']}\n")
            
            f.write("\n")
        
        f.write("---\n")
        f.write("*此报告由SDV-Theta批量运行器自动生成*\n")
    
    print(f"\n📄 详细报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    print_header()
    
    # 模型配置
    models = {
        "GaussianCopula": "model/gaussian_copula_model.py",
        "CTGAN": "model/ctgan_model.py",
        "CopulaGAN": "model/copulagan_model.py",
        "TVAE": "model/tvae_model.py",
        "PAR": "model/par_model.py"
    }
    
    # 检查模型文件是否存在
    print("\n🔍 检查模型文件...")
    missing_models = []
    for model_name, script_path in models.items():
        if not os.path.exists(script_path):
            missing_models.append((model_name, script_path))
            print(f"❌ {model_name}: {script_path} 不存在")
        else:
            print(f"✅ {model_name}: {script_path}")
    
    if missing_models:
        print(f"\n⚠️ 发现 {len(missing_models)} 个模型文件缺失，程序退出")
        return False
    
    print(f"\n✅ 所有模型文件检查通过！")
    
    # 执行模型
    print(f"\n🚀 开始执行所有模型...")
    print(f"⏰ 预计总时间: 15-30 分钟")
    
    results = {}
    total_start_time = time.time()
    
    with progress_bar(total=len(models), desc="执行模型") as pbar:
        for model_name, script_path in models.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # 为不同模型设置不同的超时时间
            if "PAR" in model_name:
                timeout = 1800  # PAR序列模型需要更多时间
            elif "CTGAN" in model_name or "CopulaGAN" in model_name or "TVAE" in model_name:
                timeout = 1500  # GAN和VAE模型需要较多时间
            else:
                timeout = 900   # GaussianCopula相对较快
            
            result = run_model(script_path, model_name, timeout=timeout)
            results[model_name] = result
            
            pbar.update(1)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 生成对比报告
    print(f"\n📊 生成对比报告...")
    report_file = generate_comparison_report(results)
    
    # 最终总结
    print("\n" + "=" * 100)
    print("🎉 所有模型执行完成!")
    print("=" * 100)
    
    successful_count = sum(1 for result in results.values() if result['success'])
    print(f"✅ 成功模型: {successful_count}/5")
    print(f"⏱️  总执行时间: {total_execution_time/60:.1f} 分钟")
    print(f"📄 对比报告: {report_file}")
    
    # 输出目录信息
    print(f"\n📁 输出目录:")
    print(f"  🔹 output/gaussian_copula/ - GaussianCopula结果")
    print(f"  🔹 output/ctgan/ - CTGAN结果")
    print(f"  🔹 output/copulagan/ - CopulaGAN结果")
    print(f"  🔹 output/tvae/ - TVAE结果")
    print(f"  🔹 output/par/ - PAR结果")
    print(f"  🔹 output/graph/ - 图表文件")
    
    print("=" * 100)
    
    if successful_count == 5:
        print("🌟 所有模型都成功执行！")
        return True
    else:
        print(f"⚠️ {5-successful_count} 个模型执行失败，请查看详细报告")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 批量运行完成!")
    else:
        print("\n⚠️ 批量运行部分失败!")
        sys.exit(1) 