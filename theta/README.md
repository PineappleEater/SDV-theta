# 🚀 SDV 多模型合成数据生成项目

本项目使用 Synthetic Data Vault (SDV) 库的多种模型来生成 `th_series_data.csv` 健康数据的合成版本。

## 📁 项目结构

```
theta/
├── source_data/
│   └── th_series_data.csv          # 原始健康数据（566K行，13列）
├── gaussian_copula/
│   └── gaussian_copula_model.py    # GaussianCopula 模型实现
├── ctgan/
│   └── ctgan_model.py              # CTGAN 模型实现
├── copulagan/
│   └── copulagan_model.py          # CopulaGAN 模型实现
├── tvae/
│   └── tvae_model.py               # TVAE 模型实现
├── par/
│   └── par_model.py                # PAR 模型实现
├── output/
│   ├── gaussian_copula/            # GaussianCopula 输出结果
│   ├── ctgan/                      # CTGAN 输出结果
│   ├── copulagan/                  # CopulaGAN 输出结果
│   ├── tvae/                       # TVAE 输出结果
│   ├── par/                        # PAR 输出结果
│   └── models_comparison_report.txt # 模型对比报告
├── utils.py                        # 通用工具函数
├── run_all_models.py               # 主执行脚本
├── compare_models.py               # 模型对比分析脚本
└── README.md                       # 项目说明文档
```

## 🤖 模型介绍

### 1. GaussianCopula
- **特点**: 基于高斯Copula的经典统计模型
- **优势**: 训练速度快，稳定性好
- **适用**: 结构化数据，追求训练速度
- **采样大小**: 10,000 行

### 2. CTGAN
- **特点**: 基于生成对抗网络的深度学习模型
- **优势**: 生成质量高，能处理复杂分布
- **适用**: 追求最高质量的合成数据
- **采样大小**: 20,000 行
- **训练参数**: 100 epochs, 256维网络

### 3. CopulaGAN
- **特点**: 结合Copula统计方法和GAN的混合模型
- **优势**: 平衡训练速度和生成质量
- **适用**: 中等复杂度数据
- **采样大小**: 15,000 行
- **训练参数**: 50 epochs, 128维网络

### 4. TVAE
- **特点**: 基于变分自编码器的表格数据合成
- **优势**: 提供潜在空间表示，适合特征学习
- **适用**: 需要数据降维和特征学习
- **采样大小**: 15,000 行
- **训练参数**: 100 epochs, 128维网络

### 5. PAR
- **特点**: 基于概率自回归的序列数据合成
- **优势**: 专门处理时间序列和序列依赖关系
- **适用**: 时间序列数据，具有用户序列的健康数据
- **采样大小**: 30,000 行
- **训练参数**: 50 epochs, 序列建模

## 🚀 快速开始

### 环境要求
```bash
# 激活 sdv conda 环境
conda activate sdv

# 确保安装了必要的包
pip install sdv pandas numpy matplotlib
```

### 运行单个模型
```bash
# 进入对应模型目录并运行
cd gaussian_copula
python3 gaussian_copula_model.py

cd ../ctgan
python3 ctgan_model.py

cd ../copulagan
python3 copulagan_model.py

cd ../tvae
python3 tvae_model.py

cd ../par
python3 par_model.py
```

### 运行所有模型
```bash
# 在 theta 目录下运行
python3 run_all_models.py
```

### 对比分析结果
```bash
# 生成模型对比报告
python3 compare_models.py
```

## 📊 输出结果

每个模型会在 `output/{model_name}/` 目录下生成：

1. **合成数据文件**: `{model_name}_synthetic_data.csv`
   - 包含1000行合成数据
   - 保持原始数据的统计特征
   - 完全匿名化敏感信息

2. **摘要报告**: `{model_name}_summary.txt`
   - 训练时间统计
   - 质量评估分数
   - 数据基本统计信息

3. **综合对比报告**: `output/models_comparison_report.txt`
   - 所有模型的性能对比
   - 质量分数排名
   - 使用建议

## 📈 性能预期

| 模型 | 训练时间 | 质量分数 | 内存使用 | 推荐场景 |
|------|----------|----------|----------|----------|
| GaussianCopula | ~30秒 | 75-85% | 低 | 快速原型 |
| CopulaGAN | ~5分钟 | 80-90% | 中等 | 平衡选择 |
| TVAE | ~8分钟 | 80-88% | 中等 | 特征学习 |
| PAR | ~10分钟 | 75-85% | 中等 | 序列数据 |
| CTGAN | ~15分钟 | 85-95% | 高 | 最高质量 |

*注：实际性能依赖于硬件配置和数据复杂度*

## 🔧 自定义配置

### 修改采样大小
在各模型脚本中修改 `sample_size` 参数：
```python
df_processed = preprocess_data(df, sample_size=50000)  # 增大采样
```

### 调整模型参数
在模型创建时修改参数：
```python
# CTGAN 示例
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=200,                    # 增加训练轮数
    batch_size=1000,              # 增大批处理
    generator_dim=(512, 512),     # 增大网络维度
)
```

### 修改生成数量
```python
synthetic_data = synthesizer.sample(num_rows=5000)  # 生成更多数据
```

## 🛠️ 数据预处理

项目自动进行以下预处理：
- **时间列转换**: 将时间字符串转换为Unix时间戳
- **缺失值处理**: 删除缺失率>80%的列
- **数据类型优化**: 自动检测和转换数据类型
- **采样**: 为提高训练速度进行合理采样
- **序列数据处理**: PAR模型会按用户ID分组并按时间排序数据

## 📝 注意事项

1. **训练时间**: CTGAN训练时间较长，建议最后运行
2. **内存使用**: 大数据集可能需要调整采样大小
3. **质量评估**: 评估分数为自动计算，可能存在误差
4. **环境依赖**: 确保在正确的conda环境中运行
5. **PAR模型**: 需要DeepEcho库，如果未安装会自动跳过

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减小采样大小
   df_processed = preprocess_data(df, sample_size=5000)
   ```

2. **训练时间过长**
   ```python
   # 减少训练轮数
   epochs=50  # 替代默认的100
   ```

3. **质量评估失败**
   - 检查数据格式是否正确
   - 确保元数据创建成功

## 📚 参考资料

- [SDV 官方文档](https://docs.sdv.dev/)
- [CTGAN 论文](https://arxiv.org/abs/1907.00503)
- [TVAE 相关研究](https://arxiv.org/abs/1907.00503)

## 📞 支持

如遇问题，请检查：
1. 环境配置是否正确
2. 数据文件路径是否存在
3. 磁盘空间是否充足
4. 内存是否足够

---

🎉 **祝您使用愉快！生成高质量的合成数据！** 