# 蛋白质判别系统 (Protein Classifier)

基于深度学习的蛋白质结构自动分类与识别系统。使用图神经网络（GNN）对 PDB 格式的蛋白质结构文件进行自动分类，支持 Web 界面交互。

## 功能特性

- **深度学习模型**: 基于图神经网络（GNN）的蛋白质分类器，结合全局特征和局部图结构
- **多类别分类**: 支持酶/受体、蛋白质-蛋白质复合物、蛋白质-核酸复合物、核酸-配体复合物四大类
- **3D 分子查看器**: 集成 3Dmol.js，按需渲染蛋白质三维结构，支持多种显示模式和着色方案
- **Web 界面**: 现代化响应式前端，支持文件上传、实时预测、数据库搜索浏览(可点击行查看详情)和3D可视化
- **中文翻译**: 自动将蛋白质分类和名称翻译为中文
- **数据库存储**: SQLite 数据库自动存储识别结果，支持查询和浏览

## 项目结构

```
protein_classifier_project/
├── README.md                    # 用户文档 (本文件)
├── MODEL_GUIDE.md               # 完整模型说明书 (零基础→复现, 721行)
├── backend/                     # 后端服务
│   ├── app.py                  # FastAPI 入口
│   ├── utils.py                # 工具函数 (翻译等)
│   ├── requirements.txt        # Python 依赖
│   ├── model/                  # 训练好的模型文件
│   │   └── protein_classifier.pth  # (~700KB, 167,748参数)
│   └── database/               # SQLite 数据库
│       ├── init_db.py          # 数据库初始化和操作
│       └── proteins.db         # 数据库文件 (39,224条)
├── frontend/                   # 前端界面
│   ├── index.html              # 主页面
│   ├── style.css               # 样式表
│   └── main.js                 # 交互逻辑
├── model_training/             # 模型训练
│   ├── train.py                # 训练脚本 (含预测接口)
│   └── prepare_data.py         # 数据准备工具
├── import_dataset.py           # 批量导入数据集到数据库
├── enrich_proteins.py          # 从RCSB PDB API获取名称
└── data/                       # 数据目录
    ├── uploads/                # 上传文件存储
    └── ...                     # 数据集目录 (P-L, P-P, P-NA, NA-L, v2016-2019, PLANET)
```

## 快速开始

### 1. 环境准备

```bash
# 安装后端依赖
cd protein_classifier_project/backend
pip install -r requirements.txt

# 安装训练依赖
cd ../model_training
pip install -r requirements.txt
```

### 2. 准备训练数据

数据来源于 PDBbind v2016–v2020 和 PDBbind-PLANET 数据库。

```bash
# 解压所有数据集到 data/ 目录
cd /d/study/AAArobot
tar -xzf P-L.tar.gz -C data/
tar -xzf P-P.tar.gz -C data/
tar -xzf P-NA.tar.gz -C data/
tar -xzf NA-L.tar.gz -C data/
tar -xzf pdbbind_v2016.tar.gz -C data/
tar -xzf pdbbind_v2017.tar.gz -C data/
tar -xzf pdbbind_v2018.tar.gz -C data/
tar -xzf pdbbind_v2019.tar.gz -C data/
tar -xzf PLANET_dataset.tar.gz -C data/
```

### 3. 训练模型

```bash
cd protein_classifier_project/model_training
python train.py
```

训练完成后，模型文件自动保存到 `backend/model/protein_classifier.pth`。

### 4. 启动后端服务

```bash
cd protein_classifier_project/backend
python app.py
```

服务启动在 http://localhost:8000

- API 文档: http://localhost:8000/docs
- 前端界面: http://localhost:8000/static/index.html

### 5. 打开前端

在浏览器中访问 http://localhost:8000/static/index.html，即可使用 Web 界面：
- 上传 PDB 文件进行预测
- 搜索已识别的蛋白质
- 浏览数据库记录

## API 接口

### POST /predict

上传 PDB 文件进行蛋白质分类预测。

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@example.pdb"
```

返回示例：
```json
{
  "name": "结合蛋白_1a07",
  "category": "酶/受体 (Protein-Ligand)",
  "category_cn": "酶/受体",
  "category_id": 0,
  "confidence": 0.9234,
  "pdb_id": "1a07",
  "sequence_length": 285,
  "all_probabilities": {
    "酶/受体": 0.9234,
    "蛋白质-蛋白质复合物": 0.0456,
    "蛋白质-核酸复合物": 0.0210,
    "核酸-配体复合物": 0.0100
  }
}
```

### GET /search

搜索蛋白质数据库。

```bash
curl "http://localhost:8000/search?q=1a07&category=酶/受体"
```

### GET /proteins

获取蛋白质列表。

```bash
curl "http://localhost:8000/proteins?limit=50&offset=0"
```

### GET /protein/{id}/pdb

获取蛋白质的原始 PDB 文件内容，供 3D 可视化使用。

```bash
curl "http://localhost:8000/protein/100/pdb"
```

返回示例：
```json
{
  "protein_id": 100,
  "pdb_id": "1aq1",
  "name": "HUMAN CYCLIN DEPENDENT KINASE 2...",
  "pdb_content": "HEADER    1AQ1_PROTEIN...",
  "file_size": 404919
}
```

### GET /stats

获取数据库统计信息。

```bash
curl http://localhost:8000/stats
```

## 模型说明

### 架构

- **类型**: 图神经网络 (Graph Neural Network, GNN)
- **节点特征**: 氨基酸类型 (one-hot 编码, 23维)
- **边特征**: 基于 Cα 原子空间距离 (8Å 阈值)
- **全局特征**: 氨基酸组成 (23维) + 物化性质 (4维) + 空间特征 (6维) + 二肽组成 (400维) = 433维
- **图卷积层**: 3层，隐藏维度 128
- **分类头**: 全连接层 (256 → 128 → 64 → 4)

### 特征提取

1. **序列特征**: 20种标准氨基酸 + 3种非标准氨基酸的组成频率
2. **物化特征**: 疏水性、电荷、极性、体积的均值
3. **空间特征**: 回旋半径、原子距离统计、惯性张量特征值
4. **二肽特征**: 400维二肽组成频率
5. **图结构**: 基于 Cα 原子的空间邻近图

### 模型性能

| 类别 | 准确率 | 训练样本数 |
|------|--------|-----------|
| 酶/受体 (P-L) | 93.8% | 1,414 |
| 蛋白质-蛋白质复合物 (P-P) | 21.1% | 65 |
| 蛋白质-核酸复合物 (P-NA) | 73.7% | 62 |
| 核酸-配体复合物 (NA-L) | 80.0% | 20 |
| **总体** | **88.4%** | 1,561 |

> P-P 类别受限于训练样本量（仅 ~91 个可用），精度有较大提升空间。详见 [MODEL_GUIDE.md](MODEL_GUIDE.md)。

### 调用接口

```python
from model_training.train import predict_protein

result = predict_protein("path/to/protein.pdb")
print(result['name'])        # 蛋白质名称
print(result['category_cn']) # 中文分类
print(result['confidence'])  # 置信度
```

## 数据库架构

### proteins 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 蛋白质名称 |
| category | TEXT | 分类 (英文) |
| category_cn | TEXT | 分类 (中文) |
| pdb_file_path | TEXT | PDB 文件路径 |
| pdb_id | TEXT | PDB 标识符 |
| confidence | REAL | 预测置信度 (数据集导入为NULL, 前端显示"已知分类") |
| sequence_length | INTEGER | 序列长度 |
| additional_info | TEXT | 中文描述信息 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

## 技术栈

- **后端**: Python 3.10+, FastAPI, Uvicorn
- **机器学习**: PyTorch, NumPy
- **数据库**: SQLite
- **前端**: HTML5, CSS3, JavaScript (原生), 3Dmol.js (3D分子渲染)
- **设计**: 响应式设计, CSS Grid/Flexbox

## 更多文档

- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** — 完整模型说明书，包含零基础科普、模型架构详解、训练过程、复现指南、参考资料和 FAQ
- **[CLAUDE.md](../CLAUDE.md)** — 项目开发手册，包含已知坑点、配置细节和常见任务指南

## 数据来源

### PDBbind 数据库 (v2016–v2020R1)

| 版本 | P-L 样本数 | 说明 |
|------|-----------|------|
| v2020R1 | 19,037 | 原始训练数据 |
| v2019 | ~17,600 | PDBbind 2019 版本 |
| v2018 | ~16,100 | PDBbind 2018 版本 |
| v2017 | ~14,700 | PDBbind 2017 版本 |
| v2016 | ~13,300 | PDBbind 2016 版本 |

PDBbind 标准四类数据集 (来自 v2020R1)：
- 蛋白质-配体复合物 (P-L): ~19,037
- 蛋白质-蛋白质复合物 (P-P): ~2,798
- 蛋白质-核酸复合物 (P-NA): ~1,032
- 核酸-配体复合物 (NA-L)

### PDBbind-PLANET 数据集

PLANET (Protein-Ligand Affinity NETwork) 是 PDBbind2020 的训练/验证/测试划分版本，包含 ~19,400 个蛋白质-配体复合物，用于结合亲和力预测基准测试。

### 总计

合并后去重，数据库共约 39,000 个唯一蛋白质结构。

### 注意事项

- P-L 类别样本数远多于其他三类，训练时采用类别级均衡采样和加权损失函数
- NA-L（核酸-配体）类别样本极少（~30），且核酸链缺乏标准氨基酸特征，模型对该类别的识别能力有限
- 当前模型对 P-L（93.8%）和 NA-L（80.0%）识别较好，P-NA 尚可（73.7%），P-P 精度较低（21.1%）受限于仅 65 个训练样本
- 数据库中 `confidence = NULL` 表示该记录来自数据集的已知分类（ground truth），前端显示"已知分类"标签；非 NULL 值表示模型实际预测的置信度
