#!/usr/bin/env python3
"""
图表管理工具
用于管理output/graph目录中的图表文件
"""

import os
import sys
from datetime import datetime

def list_graphs():
    """列出所有图表文件"""
    graph_dir = 'output/graph'
    
    if not os.path.exists(graph_dir):
        print("❌ graph目录不存在")
        return
    
    files = [f for f in os.listdir(graph_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("📊 graph目录中暂无图表文件")
        return
    
    print(f"📊 graph目录中的图表文件 ({len(files)}个):")
    print("-" * 60)
    
    for i, file in enumerate(sorted(files), 1):
        file_path = os.path.join(graph_dir, file)
        file_size = os.path.getsize(file_path)
        file_time = os.path.getmtime(file_path)
        time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{i:2d}. {file}")
        print(f"    大小: {file_size:,} 字节")
        print(f"    修改时间: {time_str}")
        print()

def clean_graphs():
    """清理graph目录"""
    graph_dir = 'output/graph'
    
    if not os.path.exists(graph_dir):
        print("❌ graph目录不存在")
        return
    
    files = [f for f in os.listdir(graph_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("📊 graph目录中暂无图表文件")
        return
    
    print(f"🧹 准备清理 {len(files)} 个图表文件...")
    
    for file in files:
        file_path = os.path.join(graph_dir, file)
        try:
            os.remove(file_path)
            print(f"✓ 已删除: {file}")
        except Exception as e:
            print(f"❌ 删除失败: {file} - {e}")
    
    print("✅ 清理完成")

def create_graph_dir():
    """创建graph目录"""
    graph_dir = 'output/graph'
    
    try:
        os.makedirs(graph_dir, exist_ok=True)
        print(f"✅ graph目录已创建: {graph_dir}")
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")

def show_usage():
    """显示使用方法"""
    print("📊 图表管理工具")
    print("=" * 40)
    print("使用方法:")
    print(f"  python3 {sys.argv[0]} list      # 列出所有图表")
    print(f"  python3 {sys.argv[0]} clean     # 清理所有图表")
    print(f"  python3 {sys.argv[0]} create    # 创建graph目录")
    print(f"  python3 {sys.argv[0]} help      # 显示此帮助")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_graphs()
    elif command == 'clean':
        clean_graphs()
    elif command == 'create':
        create_graph_dir()
    elif command == 'help':
        show_usage()
    else:
        print(f"❌ 未知命令: {command}")
        show_usage()

if __name__ == "__main__":
    main() 