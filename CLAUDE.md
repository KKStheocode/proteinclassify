# AAArobot — 蛋白质判别系统 (Protein Classifier)

## 项目概述

基于图神经网络(GNN)的蛋白质结构自动分类系统。输入 PDB 格式的蛋白质结构文件，自动分类为四大类并返回置信度。包含完整的 Web 前后端。

- **技术栈**: Python 3.10+ / FastAPI / PyTorch / SQLite / 原生 HTML+CSS+JS
- **模型**: 3层 GraphConv GNN, 隐藏维度128, 167,748 参数, 全局特征433维 + 图节点23维one-hot
- **数据来源**: PDBbind v2016R1–v2020R1 + PDBbind-PLANET (去重后 39,224 唯一蛋白, ~16.2万 PDB 文件)
- **模型性能**: 总体准确率 88.4%, P-L 93.8%, P-P 21.1%, P-NA 73.7%, NA-L 80.0%
- **文档**: README.md (用户文档), MODEL_GUIDE.md (完整模型说明书, 721行, 零基础友好)

## 目录结构

```
AAArobot/
├── CLAUDE.md                          # ← 本文件
├── data/                              # PDB 数据集 (~16.2万 PDB 文件)
│   ├── P-L/                           # 蛋白质-配体 v2020R1, ~2,599个
│   ├── P-P/                           # 蛋白质-蛋白质复合物, ~91个
│   ├── P-NA/                          # 蛋白质-核酸复合物, ~100个
│   ├── NA-L/                          # 核酸-配体复合物, ~30个
│   ├── v2016/                         # PDBbind v2016 (P-L), 26,570个
│   ├── v2017/                         # PDBbind v2017 (P-L), 29,472个
│   ├── v2018/                         # PDBbind v2018 (P-L), 32,252个
│   ├── v2019/                         # PDBbind v2019 (P-L), 35,306个
│   └── PLANET_dataset/                # PDBbind-PLANET (P-L), 38,884个
├── pdbbind_v2016.tar.gz               # 原始数据集压缩包 (已解压)
├── pdbbind_v2017.tar.gz
├── pdbbind_v2018.tar.gz
├── pdbbind_v2019.tar.gz
├── PLANET_dataset.tar.gz
├── protein_classifier_project/
│   ├── README.md                      # 用户文档
│   ├── MODEL_GUIDE.md                 # 完整模型说明书 (零基础→复现)
│   ├── backend/
│   │   ├── app.py                     # FastAPI 入口, 所有API路由
│   │   ├── utils.py                   # 翻译/工具函数
│   │   ├── requirements.txt           # fastapi, uvicorn, torch, numpy
│   │   ├── model/
│   │   │   └── protein_classifier.pth # 训练好的模型 (~700KB, 167,748参数)
│   │   └── database/
│   │       ├── init_db.py             # SQLite CRUD, 表定义
│   │       └── proteins.db            # 数据库文件 (39,224条)
│   ├── frontend/
│   │   ├── index.html                 # SPA 三面板: 预测/搜索/浏览, 3D查看器模态框, no-cache meta
│   │   ├── main.js                    # 前端逻辑 (翻页WINDOW=1, 3Dmol.js懒加载, null-safe DOM)
│   │   └── style.css                  # 响应式样式 (含3D查看器完整样式, 浏览行hover)
│   ├── model_training/
│   │   ├── train.py                   # 模型定义 + 训练 + ProteinPredictor
│   │   └── prepare_data.py            # 数据预处理 (参考用)
│   ├── import_dataset.py              # 批量导入数据集到DB (支持多目录、去重、批处理)
│   └── enrich_proteins.py             # 从RCSB PDB API获取名称 (10线程并发)
```

## 模型架构详解

### 特征提取流水线

从 PDB 文件提取两类特征：

**全局特征 (433维):**
| 特征 | 维度 | 来源 |
|------|------|------|
| 氨基酸组成 | 23 | 20标准AA + 3非标AA 频率分布 |
| 物化性质 | 4 | 疏水性/电荷/极性/体积 (均值) |
| 空间特征 | 6 | 回旋半径 + 距离统计(mean/std) + 惯性张量特征值(3) |
| 二肽组成 | 400 | 20×20 邻接氨基酸对频率 |

**图结构特征:**
- 节点: 每个氨基酸残基 → one-hot 编码 (23维: 20标准AA + 非标 + 核酸残基)
- 边: CA原子间距 < 8Å → 邻接矩阵 [N, N]
- 核酸残基 (DA/DC/DG/DT/A/C/G/U/T) 使用 C1'/C4' 作为骨架原子

### GNN 模型 (ProteinGNN)

```
PDB → 全局特征[433] ──→ GlobalEncoder(433→128→128) ──┐
     → 节点特征[N,23] → NodeEmbed(23→128)              │
     → 邻接矩阵[N,N]  → 3×GraphConv(128→128)          │
                        → MeanPool+MaxPool → [128] ────┤
                                                        ↓
                                              Concat[256]
                                                  ↓
                                    Classifier(256→128→64→4)
                                                  ↓
                                            Softmax → 4类概率
```

关键参数: hidden_dim=128, num_layers=3, dropout=0.3, max_seq_len=1000, spatial_cutoff=8.0Å

### 训练配置 (CONFIG)

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 128 | 隐藏层维度 |
| num_layers | 3 | 图卷积层数 |
| dropout | 0.3 | 训练时丢弃率 |
| learning_rate | 0.001 | Adam 初始学习率 |
| batch_size | 32 | 批大小 |
| num_epochs | 100 | 最大训练轮数 |
| early_stop_patience | 15 | 验证准确率不提升则停 |
| pdb_samples_per_class | 2000 | 每类最多采样数 |
| train/val/test ratio | 70/15/15 | 数据划分比例 |
| random_seed | 42 | 固定随机种子 |

**类别均衡策略:**
1. 类别级采样: `prepare_data()` 先收集每类所有目录的PDB文件, 再统一采样至 `pdb_samples_per_class`
2. 加权损失: `CrossEntropyLoss(weight=class_weights)`, 权重 = total/(4×count)
   - P-L (1414): 0.28, P-P (65): 6.00, P-NA (62): 6.29, NA-L (20): 19.51

### NA-L (核酸-配体) 特殊处理

NA-L 文件的核酸残基主导, 默认的 `parse_pdb_structure` 会因 `aa_count > na_count` 失败将其过滤。做了以下修改:

1. `parse_pdb_structure(filepath, allow_na=True)` — 当 `allow_na=True` 时, 只要求 `(aa_count + na_count) > 3`
2. 核酸使用 C1'/C4' 作为骨架参考原子 (替代 CA)
3. `AA_INDEX['X'] = 20` — 核酸残基映射到第20维 (未知/非标准)
4. `AA_PROPERTIES['X']` — 中性默认值 [0,0,0,150]
5. `extract_dipeptide_features` — 仅统计 `a1<20 and a2<20` 的标准AA配对

## 数据库 (SQLite)

表 `proteins` 字段: id, name, category (英文全称), category_cn (中文简称), pdb_file_path, pdb_id, confidence, sequence_length, additional_info (中文介绍), created_at, updated_at

四类分类及其数据量:
| ID | category | category_cn | 数据库记录数 | 训练可用样本 |
|----|----------|-------------|-------------|-------------|
| 0 | 酶/受体 (Protein-Ligand) | 酶/受体 | ~39,000 | ~2000 (采样) |
| 1 | 蛋白质-蛋白质复合物 (Protein-Protein) | 蛋白质-蛋白质复合物 | ~91 | ~100 |
| 2 | 蛋白质-核酸复合物 (Protein-Nucleic Acid) | 蛋白质-核酸复合物 | ~100 | ~100 |
| 3 | 核酸-配体复合物 (Nucleic Acid-Ligand) | 核酸-配体复合物 | ~30 | ~30 |

### 数据集与目录映射 (import_dataset.py)

```python
CATEGORY_MAP = {
    'P-L':                          → 酶/受体,
    'v2016':                        → 酶/受体,
    'v2017':                        → 酶/受体,
    'v2018':                        → 酶/受体,
    'v2019':                        → 酶/受体,
    'PLANET_dataset/PDBbind2020-PLANET' → 酶/受体,
    'P-P':  → 蛋白质-蛋白质复合物,
    'P-NA': → 蛋白质-核酸复合物,
    'NA-L': → 核酸-配体复合物,
}
```

### 训练数据目录映射 (train.py CONFIG['data_dirs'])

```python
0: ['../data/P-L/', '../data/v2016/', '../data/v2017/',
    '../data/v2018/', '../data/v2019/',
    '../data/PLANET_dataset/PDBbind2020-PLANET/'],
1: '../data/P-P/',
2: '../data/P-NA/',
3: '../data/NA-L/',
```

> 注意: train.py 支持列表值(多目录合并到一个类别), `prepare_data()` 会先收集所有目录的 PDB 文件再按类别统一采样。

## 模型性能详情

训练数据: 2,230 样本 (P-L:2000, P-P:100, P-NA:100, NA-L:30)
训练集/验证集/测试集: 1561/334/335

| 类别 | 测试准确率 | 正确/总数 | 训练样本数 | 权重 |
|------|-----------|-----------|-----------|------|
| 酶/受体 (P-L) | 93.8% | 274/292 | 1414 | 0.28 |
| 蛋白质-蛋白质 (P-P) | 21.1% | 4/19 | 65 | 6.00 |
| 蛋白质-核酸 (P-NA) | 73.7% | 14/19 | 62 | 6.29 |
| 核酸-配体 (NA-L) | 80.0% | 4/5 | 20 | 19.51 |
| **总体** | **88.4%** | 296/335 | 1561 | - |

**混淆矩阵特征:**
- P-P 最易与 P-L 混淆 (10/19 误判为P-L) — 两者都是蛋白质为主的结构
- P-NA 和 NA-L 被正确识别时置信度较高
- P-L 少量被误判为 P-P (12/292) 和 P-NA (5/292)

**性能瓶颈:** P-P 类仅 65 训练样本, 深度学习需要更多数据。如需提升 P-P 精度, 需要额外数据源 (如 Docking Benchmark、BioLiP)。

## 重要实现细节 (含已知坑点)

### 1. train.py CONFIG 不允许重复键
Python dict 重复键取最后一个值。之前 `pdb_samples_per_class` 出现两次 (2000 和 500), 最终使用 500。**修改 CONFIG 时检查是否有重复键。**

### 2. 类别权重计算的 tensor hash bug
```python
# ❌ 错误: tensor 作为 dict key, 每个 tensor 的 hash 不同
label_counts = defaultdict(int)
for _, label, _, _, _ in train_set:
    label_counts[label] += 1  # 每样本一个 key, 权重全相等!

# ✅ 正确: 转为 Python int
label_counts = [0, 0, 0, 0]
for _, label, _, _, _ in train_set:
    label_counts[int(label)] += 1
```

### 3. NA-L 文件解析
NA-L 的 PDB 文件中核酸残基占多数, `parse_pdb_structure` 默认过滤条件 `aa_count > na_count` 会丢弃所有核酸链。必须用 `allow_na=True`, 且核酸残基使用 C1'/C4' 作为骨架原子。`extract_features` 和 `ProteinPredictor.predict` 都调用了 `allow_na=True`。

### 4. 搜索功能的分类过滤
前端传入中文分类名(如"酶/受体"), 后端在 `category_cn` 列用 `=` 精确匹配。`CATEGORY_TRANSLATIONS` 字典是双向映射 — `app.py:196-204` 有专门处理。

### 5. 前端翻页
数据量 39,224 条, 每页 20 条 = ~1,962 页。`renderPagination()` 滑动窗口 WINDOW=1 (前后各1页), 超出用省略号。有页码跳转输入框(回车触发)和上/下一页按钮。`overflow-x: auto` 兜底防止溢出。

### 6. 蛋白质名称来源
数据集中的 PDBbind 文件不含 TITLE 头。名称通过 RCSB PDB REST API (`https://data.rcsb.org/rest/v1/core/entry/{pdb_id}`) 获取。`enrich_proteins.py` 使用 10 线程并发, API 有限流 (~50% 成功率, 约半数用默认名)。

### 7. 数据库中置信度的含义
- 数据集导入的记录 (ground truth) → `confidence = NULL` → 前端显示"已知分类"灰色标签
- 通过 `/predict` API 预测的记录 → 有实际的模型置信度值 (0~1) → 前端显示百分比
- `import_dataset.py` 中 confidence 参数为 `None`, 不要设为 `1.0` (会误导为模型100%确信)

### 8. PDB 文件名模式
- P-L (所有版本): `{pdb_id}_protein.pdb` (在 `{group}/{pdb_id}/` 子目录)
- PLANET: `{pdb_id}_protein.pdb` (在 `PDBbind2020-PLANET/{pdb_id}/` 子目录)
- P-P: `{pdb_id}_complex.pdb` (扁平目录)
- P-NA: `{pdb_id}_complex.pdb` (扁平目录)
- NA-L: `{pdb_id}_nucleic_acid.pdb` (在 `{pdb_id}/` 子目录)

### 9. 浏览器缓存与 displayResult 防护
前端的 `displayResult()` 函数操作多个 DOM 元素（`#resultCard`, `#confidenceBadge`, `#btnView3D` 等）。如果浏览器缓存了旧版 HTML（缺少某个元素），直接访问 `.dataset` 或 `.textContent` 会抛出 TypeError，被 `showProteinDetail` 的 catch 捕获后弹出"获取蛋白质详情失败"。

**当前防护:**
- `displayResult` 中所有 `$()` 调用均加 null 检查后再访问属性
- `showProteinDetail` 中 `displayResult` 调用包裹在独立 try-catch 中, 抛错不冒泡到外层 toast
- `showProteinDetail` 中 `switchTab('predict')` 在 `displayResult` 之前执行, 确保面板已激活
- `index.html` 添加了 no-cache meta 标签 (但不保证所有浏览器遵守)

**如果仍然报错**: 让用户在浏览器中 Ctrl+Shift+R 强制刷新, 清除缓存的旧 HTML/JS/CSS。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 + 模型状态 |
| POST | `/predict` | 上传 PDB 文件预测 (multipart form, field: `file`) |
| GET | `/search?q=&category=&limit=` | 搜索蛋白质 (q=关键词, category=中文分类) |
| GET | `/proteins?limit=&offset=` | 分页浏览所有蛋白质 |
| GET | `/protein/{id}` | 获取单条蛋白质详情 |
| GET | `/protein/{id}/pdb` | 获取蛋白质 PDB 原始文件内容 (用于3D可视化) |
| GET | `/stats` | 数据库统计 (总数+分类分布) |

## 预测模型调用

```python
from model_training.train import ProteinPredictor

predictor = ProteinPredictor('path/to/protein_classifier.pth')
result = predictor.predict('path/to/protein.pdb')
# result: {name, category, category_cn, category_id, confidence,
#          pdb_id, sequence_length, all_probabilities}
```

## 启动方式

```bash
# 安装依赖 (首次)
pip install fastapi uvicorn python-multipart torch numpy requests

# 启动后端 (从项目根目录)
cd protein_classifier_project/backend
python app.py
```

- API: http://localhost:8000
- API文档(Swagger): http://localhost:8000/docs
- 前端: http://localhost:8000/static/index.html

## 已完成的重大操作

1. **数据集导入** (39,224条): `import_dataset.py` 扫描 9 个目录, pdb_id 去重, 批量写入 SQLite (500条/批)
2. **名称富集** (19,682/39,226): `enrich_proteins.py` 10线程从 RCSB PDB API 获取, 其余用默认名
3. **多版本数据合并**: 添加 PDBbind v2016–v2019 + PDBbind-PLANET (均为 P-L 类)
4. **类别均衡训练**: 按类别统一采样 (非按目录), 加权 CrossEntropyLoss
5. **NA-L 解析修复**: `allow_na=True` + C1'/C4'骨架 + AA_INDEX['X'] + AA_PROPERTIES['X']
6. **权重计算 bug 修复**: tensor hash → Python int, 类别权重现在正确生效
7. **搜索修复**: 分类过滤从 LIKE 改为精确 category_cn 匹配
8. **翻页优化**: 全量渲染 → 滑动窗口(WINDOW=1) + 跳转输入 + prev/next, overflow-x 兜底
9. **置信度修复**: 数据集导入 confidence 从 1.0 改为 NULL, 前端区分显示"已知分类" vs 模型置信度百分比
10. **模型说明书**: 编写 MODEL_GUIDE.md (721行, 涵盖零基础科普到复现指南)
11. **3D 蛋白质查看器**: 集成 3Dmol.js, 按需渲染 PDB 结构, 模态框展示, 支持 4 种显示模式 + 4 种着色方案, 关闭时彻底销毁 WebGL 上下文, 错误恢复机制
12. **前端鲁棒性增强**: displayResult 全部 DOM 访问加 null 检查, showProteinDetail 嵌套 try-catch 防止 displayResult 崩溃弹错误, switchTab 提前到 displayResult 之前, 浏览面板行可点击跳转详情

## 常见任务指南

### 重启后端
```bash
taskkill //F //IM python.exe   # Windows 杀掉所有 python
cd protein_classifier_project/backend && python app.py
```
服务配置 `reload=True`, 修改 `backend/` 下的 Python 文件会自动重启。

### 重新训练模型
```bash
cd protein_classifier_project/model_training
python train.py
```
训练数据来自 `CONFIG['data_dirs']`, 模型保存到 `backend/model/protein_classifier.pth`。

### 添加新的数据集蛋白质
```bash
cd protein_classifier_project
python import_dataset.py    # 增量导入 (pdb_id 去重, 跳过已存在)
python enrich_proteins.py   # 从 RCSB API 补充名称和介绍
```

### 添加新数据集目录
1. 解压到 `data/` 下
2. 在 `import_dataset.py` 的 `CATEGORY_MAP` 添加目录→类别映射
3. 在 `train.py` 的 `CONFIG['data_dirs']` 对应类别列表添加目录
4. 重新导入和训练

### 添加新分类
1. `train.py`: `CATEGORY_NAMES` + `CATEGORY_NAMES_CN` 添加
2. `utils.py`: `CATEGORY_TRANSLATIONS` + `CATEGORY_INFO_CN` 添加
3. `import_dataset.py`: `CATEGORY_MAP` + `NAME_PREFIX` 添加
4. `index.html`: `<select id="categoryFilter">` 添加选项
5. 重新训练模型 (num_classes 需修改)

## 环境

- OS: Windows 11
- Shell: bash (Git Bash), 路径用 Unix 风格 `/`
- Python: Anaconda, 依赖在 `D:\Anaconda\Lib\site-packages\`
- 工作目录: `D:\study\AAArobot`
- GPU: CUDA available (训练时自动使用)
