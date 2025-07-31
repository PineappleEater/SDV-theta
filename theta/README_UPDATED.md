# SDV-Theta 项目 - 整理版

## 📋 项目概述

这是一个基于SDV（Synthetic Data Vault）的健康数据合成项目，包含5个不同的合成数据模型和用户未来数据预测功能。

## 🗂️ 文件结构

### 主要功能文件

| 文件名 | 功能描述 | 使用场景 |
|-------|---------|---------|
| **`comprehensive_evaluator.py`** ⭐ | 综合数据评测器 | **推荐使用** - 功能最强大的评测脚本 |
| `run_all_models.py` | 批量运行所有模型 | 一次性对比5个SDV模型的效果 |
| `compare_models.py` | 模型对比工具 | 详细对比不同模型的性能 |
| `utils.py` | 核心工具函数库 | 包含改进的数据预处理功能 |
| `manage_graphs.py` | 图表管理工具 | 管理output/graph目录中的图表文件 |

### 模型文件
```
model/
├── gaussian_copula_model.py    # 高斯Copula模型
├── ctgan_model.py             # CTGAN模型  
├── copulagan_model.py         # CopulaGAN模型
├── tvae_model.py              # TVAE模型
└── par_model.py               # PAR序列模型
```

### 数据文件
```
source_data/
├── th_series_data.csv         # 主要健康时间序列数据
├── th_series_dim.csv          # 维度数据
└── th_series_dim_no_embedding.csv
```

### 输出文件
```
output/
├── gaussian_copula/           # GaussianCopula输出
├── ctgan/                     # CTGAN输出
├── copulagan/                 # CopulaGAN输出
├── tvae/                      # TVAE输出
├── par/                       # PAR输出
├── user_predictions/          # 用户预测结果（简单版）
├── comprehensive_reports/     # 📋 综合评测报告和数据
└── graph/                     # 📊 所有图表文件统一保存位置
    ├── user_*_comprehensive_analysis.png      # 综合分析图表
    └── user_*_top16_indicators_analysis.png   # 🆕 前16个指标详细分析
```

## 🚀 快速开始

### 1. 运行所有SDV模型（训练和生成数据）

```bash
# 批量运行5个模型进行训练和数据生成
python3 run_all_models.py
```

### 2. 综合数据评测（分析和报告生成）

```bash
# 查看使用方法和可用用户
python3 comprehensive_evaluator.py

# 分析用户169的所有可用模型数据
python3 comprehensive_evaluator.py 169

# 只分析特定模型（CTGAN和PAR）
python3 comprehensive_evaluator.py 169 ctgan,par

# 分析所有模型，限制最多5000条记录
python3 comprehensive_evaluator.py 169 all 5000
```

**🔄 推荐工作流程**：
1. 📊 首先运行 `run_all_models.py` 训练模型并生成合成数据
2. 📈 然后运行 `comprehensive_evaluator.py` 进行分析和报告生成

**📋 综合评测器功能**：
- ✅ 🆕 **多模型对比分析**：同时分析5个模型的合成数据
- ✅ 用户数据预处理和分析
- ✅ 统计分析与原数据对比  
- ✅ 生成6张综合分析图表
- ✅ **🆕 前16个指标详细分析**：4x4网格分布对比图
- ✅ 生成完整的Markdown分析报告
- ✅ 包含数据结果实例和质量评估

**🆕 新增：前16个指标详细分析**
- 📊 自动分析用户最常见的16个健康指标
- 📈 生成原数据vs合成数据的分布对比图（4x4网格）
- 📋 数值型指标：显示均值、标准差、分布相似性
- 📋 分类型指标：显示值频率分布和覆盖率
- 📄 在报告中包含每个指标的详细统计对比

### 3. 运行单个模型（如需要）

```bash
# 运行高斯Copula模型（推荐新手）
python3 model/enhanced_gaussian_copula_model.py

# 运行CTGAN模型（质量最高）
python3 model/enhanced_ctgan_model.py
```

## 🔧 核心改进

### 1. 数据预处理优化

**改进前的问题：**
- indicator列56.24%的数据被归为"other"
- CTGAN等模型质量分数仅50%左右
- 时间列处理不当导致高基数问题

**改进后的效果：**
- **frequency_based策略**：indicator"other"比例降至1.1%
- **智能数值转换**：value列74%+成功转换为数值
- **时间特征提取**：转换时间为hour/day_of_week/month特征
- **质量分数提升**：GaussianCopula从53%提升到77%

### 2. 预处理策略选择

```python
# 在utils.py中的preprocess_data函数支持3种策略：

# 推荐：基于频率的策略（效果最佳）
preprocess_data(df, strategy='frequency_based')  

# 自适应策略（平衡覆盖率）
preprocess_data(df, strategy='adaptive')

# 简单策略（原方法）
preprocess_data(df, strategy='simple')
```

## 📊 模型性能对比

| 模型 | 质量分数 | 训练时间 | 特点 | 推荐场景 |
|------|---------|---------|------|---------|
| **GaussianCopula** | 77.04% | ~1秒 | 快速稳定 | 快速验证、生产环境 |
| **CTGAN** | 78%+ | ~140秒 | 质量最高 | 追求极致效果 |
| **CopulaGAN** | 70%+ | ~90秒 | 平衡性好 | 综合应用 |
| **TVAE** | 65%+ | ~20秒 | 中等性能 | 特征学习 |
| **PAR** | 60%+ | ~900秒 | 序列专用 | 时间序列数据 |

## 💡 使用建议

### 新手推荐流程

1. **综合评测** ⭐ **（首选）**：
   ```bash
   python3 comprehensive_evaluator.py 169 7
   ```

2. **快速体验单个模型**：
   ```bash
   python3 model/gaussian_copula_model.py
   ```

3. **全面模型对比**：
   ```bash
   python3 run_all_models.py
   ```

### 生产环境推荐

- **快速训练**：GaussianCopula（1秒训练，77%质量）
- **最佳质量**：CTGAN（140秒训练，78%+质量）
- **序列数据**：PAR（专门处理时间序列）

## 🔍 核心技术特性

### 数据预处理改进

1. **智能indicator处理**：
   - frequency_based策略保留出现≥3次的指标
   - 解决了"other"占比过高的问题

2. **智能value处理**：
   - 自动检测数值转换成功率
   - 74%+成功率时进行数值化
   - 否则保留前100个常见值

3. **时间特征工程**：
   - 提取hour、day_of_week、month特征
   - 避免原始时间戳的高基数问题

### 用户预测系统

1. **个性化建模**：基于单个用户历史数据训练PAR模型
2. **智能用户筛选**：自动识别数据量适中的用户（200-10,000条记录）
3. **时间序列生成**：生成未来1-365天的个性化健康数据
4. **结果分析**：自动生成预测报告和数据分析

## 🎯 项目亮点

1. **质量大幅提升**：通过改进预处理，模型质量分数提升20-30个百分点
2. **高度可配置**：支持多种预处理策略和参数调整
3. **完整工作流**：从数据加载到结果分析的端到端解决方案
4. **用户友好**：简单的命令行接口和详细的使用文档
5. **生产就绪**：包含错误处理、超时机制和质量验证

## ⚠️ 注意事项

1. **内存使用**：大用户数据（>10,000条记录）可能导致内存不足
2. **训练时间**：PAR模型训练时间较长，建议使用较小的epochs参数
3. **数据质量**：预测效果依赖于用户历史数据的质量和完整性

## 📝 更新日志

- ✅ 删除了所有debug文件
- ✅ 将改进的预处理功能合并到utils.py
- ✅ 统一了所有模型的预处理策略
- ✅ 优化了"other"值的处理逻辑
- ✅ 改进了时间特征提取
- ✅ 提升了整体数据质量和模型性能 