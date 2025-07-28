#!/usr/bin/env python3
"""
主执行脚本 - 依次运行所有SDV模型进行合成数据生成
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_model(model_name, script_path):
    """运行单个模型"""
    print(f"\n{'='*80}")
    print(f"开始运行 {model_name} 模型")
    print(f"脚本路径: {script_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # 切换到模型目录并运行脚本
        model_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=model_dir,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 模型执行成功！")
            print(f"执行时间: {duration:.2f} 秒")
            print("输出:")
            print(result.stdout)
        else:
            print(f"❌ {model_name} 模型执行失败！")
            print("错误信息:")
            print(result.stderr)
            
        return result.returncode == 0, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 运行 {model_name} 时出现异常: {e}")
        return False, duration

def main():
    print("🚀 开始运行所有SDV模型进行合成数据生成")
    print(f"开始时间: {datetime.now()}")
    
    # 定义所有模型
    models = [
        ("GaussianCopula", "gaussian_copula/gaussian_copula_model.py"),
        ("CopulaGAN", "copulagan/copulagan_model.py"),
        ("TVAE", "tvae/tvae_model.py"),
        ("PAR", "par/par_model.py"),      # PAR序列模型
        ("CTGAN", "ctgan/ctgan_model.py")  # CTGAN放最后，因为训练时间最长
    ]
    
    results = {}
    total_start_time = time.time()
    
    for model_name, script_path in models:
        success, duration = run_model(model_name, script_path)
        results[model_name] = {
            'success': success,
            'duration': duration
        }
        
        # 在模型之间稍作停顿
        if model_name != models[-1][0]:  # 不是最后一个模型
            print(f"\n⏳ 等待3秒后运行下一个模型...")
            time.sleep(3)
    
    # 打印总结报告
    total_duration = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("📊 执行总结报告")
    print(f"{'='*80}")
    print(f"总执行时间: {total_duration:.2f} 秒")
    print(f"结束时间: {datetime.now()}")
    
    print(f"\n各模型执行结果:")
    for model_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"  {model_name:15} | {status} | 耗时: {result['duration']:.2f}s")
    
    # 统计成功率
    successful_models = sum(1 for r in results.values() if r['success'])
    total_models = len(results)
    success_rate = (successful_models / total_models) * 100
    
    print(f"\n成功执行: {successful_models}/{total_models} 个模型 ({success_rate:.1f}%)")
    
    if successful_models > 0:
        print(f"\n📁 所有结果已保存到 output/ 目录下的对应子文件夹中")
        print(f"可以运行 python3 compare_models.py 来对比不同模型的结果")
    
    print(f"\n🎉 所有模型执行完成！")

if __name__ == "__main__":
    main() 