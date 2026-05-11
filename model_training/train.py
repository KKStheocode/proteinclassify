"""
蛋白质判别模型训练脚本
功能：从 PDB 文件提取特征，训练 GNN 分类器，输出可调用的 predict_protein 函数
类别：酶/受体(Protein-Ligand), 蛋白质-蛋白质复合物, 蛋白质-核酸复合物, 核酸-配体
"""

import os
import sys
import re
import gzip
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    # 数据路径 (相对于项目根目录 protein_classifier_project/)
    'data_dirs': {
        0: [  # Protein-Ligand → Enzyme/Receptor (多版本合并)
            '../data/P-L/',
            '../data/v2016/',
            '../data/v2017/',
            '../data/v2018/',
            '../data/v2019/',
            '../data/PLANET_dataset/PDBbind2020-PLANET/',
        ],
        1: '../data/P-P/',   # Protein-Protein Complex
        2: '../data/P-NA/',  # Protein-Nucleic Acid Complex
        3: '../data/NA-L/',  # Nucleic Acid-Ligand
    },
    'pdb_samples_per_class': 2000,  # 每类最多采样数 (数据量增加)
    # 模型参数
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'early_stop_patience': 15,
    # 特征参数
    'max_seq_len': 1000,
    'num_aa_types': 23,  # 20 standard + 3 non-standard
    'spatial_cutoff': 8.0,  # Å, for graph edge construction
    'max_neighbors': 30,
    # 数据处理
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
}

CATEGORY_NAMES = {
    0: '酶/受体 (Protein-Ligand)',
    1: '蛋白质-蛋白质复合物 (Protein-Protein)',
    2: '蛋白质-核酸复合物 (Protein-Nucleic Acid)',
    3: '核酸-配体复合物 (Nucleic Acid-Ligand)',
}

CATEGORY_NAMES_CN = {
    0: '酶/受体',
    1: '蛋白质-蛋白质复合物',
    2: '蛋白质-核酸复合物',
    3: '核酸-配体复合物',
}

# ============================================================
# PDB 解析与特征提取
# ============================================================

# 标准氨基酸三字母码 → 单字母码
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # 非标准氨基酸 (常见)
    'MSE': 'M', 'HSE': 'H', 'HSD': 'H', 'HSP': 'H', 'HIE': 'H',
    'HIP': 'H', 'CSE': 'C', 'CSD': 'C', 'CSX': 'C', 'SEC': 'C',
    'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'KCX': 'K', 'LLP': 'K',
}

# 20种标准氨基酸的特征索引
AA_INDEX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # 非标准氨基酸 / 核酸残基
}

# 氨基酸物化性质 (用于增强特征)
AA_PROPERTIES = {
    'A': [1.8, 0.0, 0.0, 89.1],   # hydrophobicity, charge, polarity, volume
    'C': [2.5, 0.0, 0.0, 108.5],
    'D': [-3.5, -1.0, 1.0, 111.1],
    'E': [-3.5, -1.0, 1.0, 138.4],
    'F': [2.8, 0.0, 0.0, 189.9],
    'G': [-0.4, 0.0, 0.0, 60.1],
    'H': [-3.2, 0.5, 1.0, 153.2],
    'I': [4.5, 0.0, 0.0, 166.7],
    'K': [-3.9, 1.0, 1.0, 168.6],
    'L': [3.8, 0.0, 0.0, 166.7],
    'M': [1.9, 0.0, 0.0, 162.9],
    'N': [-3.5, 0.0, 1.0, 114.1],
    'P': [-1.6, 0.0, 0.0, 112.7],
    'Q': [-3.5, 0.0, 1.0, 143.8],
    'R': [-4.5, 1.0, 1.0, 173.4],
    'S': [-0.8, 0.0, 1.0, 89.0],
    'T': [-0.7, 0.0, 1.0, 116.1],
    'V': [4.2, 0.0, 0.0, 140.0],
    'W': [-0.9, 0.0, 0.0, 227.8],
    'Y': [-1.3, 0.0, 1.0, 193.6],
    'X': [0.0, 0.0, 0.0, 150.0],  # 未知/核酸残基 (中性值)
}

# 核酸残基名称识别
NA_RESIDUES = {'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'U', 'T',
               'RA', 'RC', 'RG', 'RU', 'ADE', 'CYT', 'GUA', 'THY', 'URA'}


def read_pdb(filepath):
    """读取 PDB 文件，支持 .pdb 和 .pdb.gz"""
    if str(filepath).endswith('.gz'):
        with gzip.open(filepath, 'rt', errors='ignore') as f:
            return f.readlines()
    else:
        with open(filepath, 'r', errors='ignore') as f:
            return f.readlines()


def parse_pdb_structure(filepath, allow_na=False):
    """
    解析 PDB 文件，提取蛋白质链信息
    Args:
        filepath: PDB 文件路径
        allow_na: 是否允许核酸链 (NA-L 类别需要)
    返回:
        chains: dict {chain_id: [residues]}
    """
    lines = read_pdb(filepath)
    chains = defaultdict(list)
    current_res = None
    current_chain = None
    seen_res = set()

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            res_seq = line[22:26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            alt_loc = line[16:17].strip()

            if alt_loc and alt_loc != 'A':
                continue

            res_key = (chain_id, res_seq, res_name)
            if res_key != current_res or chain_id != current_chain:
                current_res = res_key
                current_chain = chain_id
                if res_key not in seen_res:
                    seen_res.add(res_key)
                    chains[chain_id].append({
                        'resname': res_name,
                        'resid': res_seq,
                        'atoms': [],
                        'ca_coord': None,
                    })

            if chains[chain_id]:
                chains[chain_id][-1]['atoms'].append(
                    (atom_name, x, y, z, element)
                )
                if atom_name == 'CA':
                    chains[chain_id][-1]['ca_coord'] = np.array([x, y, z])
                # 核酸使用 C1' 或 C4' 作为骨架原子
                if allow_na and atom_name in ("C1'", "C4'"):
                    chains[chain_id][-1]['ca_coord'] = np.array([x, y, z])

    # 过滤链
    filtered = {}
    for chain_id, residues in chains.items():
        aa_count = sum(1 for r in residues if r['resname'] in AA_3TO1)
        na_count = sum(1 for r in residues if r['resname'] in NA_RESIDUES)
        if allow_na:
            # NA-L 模式: 接受任何足够长的链
            total = aa_count + na_count
            if total > 3:
                filtered[chain_id] = residues
        else:
            if aa_count > na_count and aa_count > 3:
                filtered[chain_id] = residues

    return filtered


def get_sequence_and_coords(chains):
    """从解析的链中提取序列和 CA 坐标"""
    sequence = ''
    ca_coords = []
    residue_names = []

    for chain_id in sorted(chains.keys()):
        for res in chains[chain_id]:
            aa1 = AA_3TO1.get(res['resname'], 'X')
            if aa1 in AA_INDEX:
                sequence += aa1
                residue_names.append(aa1)
                if res['ca_coord'] is not None:
                    ca_coords.append(res['ca_coord'])
                else:
                    ca_coords.append(np.zeros(3))

    return sequence, ca_coords, residue_names


def extract_composition_features(sequence, residue_names):
    """提取氨基酸组成特征 (23维，含非标准)"""
    features = np.zeros(CONFIG['num_aa_types'])

    for r in residue_names:
        if r in AA_INDEX:
            features[AA_INDEX[r]] += 1

    total = len(residue_names)
    if total > 0:
        features = features / total

    return features


def extract_physicochemical_features(residue_names):
    """提取物化性质特征 (4维均值)"""
    props = np.zeros(4)
    count = 0
    for r in residue_names:
        if r in AA_PROPERTIES:
            props += np.array(AA_PROPERTIES[r])
            count += 1
    if count > 0:
        props /= count
    return props


def extract_spatial_features(ca_coords):
    """提取空间特征"""
    if len(ca_coords) < 2:
        return np.zeros(6)

    coords = np.array(ca_coords)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    # 回旋半径
    rg = np.sqrt(np.mean(distances ** 2))

    # 惯性张量特征值 (描述形状)
    centered = coords - centroid
    if len(coords) >= 3:
        try:
            tensor = np.dot(centered.T, centered) / len(coords)
            eigenvalues = np.linalg.eigvalsh(tensor)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # 归一化
            if eigenvalues[0] > 0:
                eigenvalues = eigenvalues / eigenvalues[0]
        except np.linalg.LinAlgError:
            eigenvalues = np.array([1.0, 0.5, 0.1])
    else:
        eigenvalues = np.array([1.0, 0.5, 0.1])

    features = np.concatenate([
        [rg],
        [np.mean(distances), np.std(distances)],
        eigenvalues,
    ])
    return features


def extract_dipeptide_features(sequence):
    """提取二肽组成特征 (400维)"""
    dipep = np.zeros(400)
    valid_pairs = 0
    for i in range(len(sequence) - 1):
        a1 = AA_INDEX.get(sequence[i], -1)
        a2 = AA_INDEX.get(sequence[i+1], -1)
        if a1 < 20 and a2 < 20:  # 仅统计标准氨基酸配对
            idx = a1 * 20 + a2
            dipep[idx] += 1
            valid_pairs += 1
    if valid_pairs > 0:
        dipep = dipep / valid_pairs
    return dipep


def extract_features(filepath):
    """
    从 PDB 文件提取完整特征向量
    返回: np.array (特征向量), str (序列), dict (元数据)
    """
    try:
        chains = parse_pdb_structure(filepath, allow_na=True)
        if not chains:
            return None, None, None

        sequence, ca_coords, residue_names = get_sequence_and_coords(chains)

        if len(residue_names) < 10:  # 太小的结构跳过
            return None, None, None

        # 截断长序列
        if len(residue_names) > CONFIG['max_seq_len']:
            residue_names = residue_names[:CONFIG['max_seq_len']]
            sequence = sequence[:CONFIG['max_seq_len']]
            ca_coords = ca_coords[:CONFIG['max_seq_len']]

        comp_feat = extract_composition_features(sequence, residue_names)
        phys_feat = extract_physicochemical_features(residue_names)
        spatial_feat = extract_spatial_features(ca_coords)
        dipep_feat = extract_dipeptide_features(sequence)

        features = np.concatenate([comp_feat, phys_feat, spatial_feat, dipep_feat])
        meta = {'sequence': sequence, 'length': len(residue_names)}

        return features, sequence, meta
    except Exception as e:
        return None, None, None


def build_graph_adjacency(ca_coords, cutoff=None):
    """基于 CA 原子空间距离构建图邻接矩阵"""
    if cutoff is None:
        cutoff = CONFIG['spatial_cutoff']

    n = len(ca_coords)
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))

    coords = np.array(ca_coords)
    # 计算距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # 邻接矩阵 (基于距离阈值)
    adj = (dist_matrix < cutoff).astype(np.float32)
    np.fill_diagonal(adj, 0)

    # 边特征 (距离)
    edge_feat = dist_matrix.copy()
    edge_feat[adj == 0] = 0

    return adj, edge_feat


# ============================================================
# GNN 模型定义
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        # x: [batch, num_nodes, in_dim]
        # adj: [batch, num_nodes, num_nodes]
        # 归一化邻接矩阵
        D = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / D

        out = torch.bmm(adj_norm, x)
        out = self.linear(out)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class ProteinGNN(nn.Module):
    """
    蛋白质图神经网络分类器
    结合全局特征和局部图结构进行蛋白质分类
    """
    def __init__(self, global_feat_dim, num_classes=4, hidden_dim=128,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 节点特征嵌入 (将残基类型 one-hot 映射到 hidden_dim)
        self.node_embed = nn.Linear(CONFIG['num_aa_types'], hidden_dim)

        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.gcn_layers.append(
                GraphConvLayer(in_dim, hidden_dim, dropout)
            )

        # 全局特征处理
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, global_feat, node_feat, adj):
        """
        global_feat: [batch, global_feat_dim]
        node_feat: [batch, num_nodes, num_aa_types]
        adj: [batch, num_nodes, num_nodes]
        """
        batch_size = global_feat.size(0)

        # 图卷积
        x = self.node_embed(node_feat)
        for gcn in self.gcn_layers:
            x = gcn(x, adj)

        # 全局池化 (mean + max)
        x_mean = x.mean(dim=1)  # [batch, hidden_dim]
        x_max = x.max(dim=1)[0]  # [batch, hidden_dim]
        graph_feat = x_mean + x_max

        # 全局特征编码
        global_enc = self.global_encoder(global_feat)

        # 融合
        combined = torch.cat([graph_feat, global_enc], dim=1)
        out = self.classifier(combined)
        return out


class ProteinDataset(torch.utils.data.Dataset):
    """蛋白质数据集 (预提取特征)"""
    def __init__(self, features, labels, adj_matrices, node_features, names):
        self.features = features
        self.labels = labels
        self.adj_matrices = adj_matrices
        self.node_features = node_features
        self.names = names

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.labels[idx]])[0],
            torch.FloatTensor(self.adj_matrices[idx]),
            torch.FloatTensor(self.node_features[idx]),
            self.names[idx],
        )


def collate_fn(batch):
    """自定义批处理: 补齐不同大小的图"""
    features, labels, adjs, nodes, names = zip(*batch)

    # 找到最大节点数
    max_nodes = max(n.shape[0] for n in nodes)
    node_dim = nodes[0].shape[1]

    padded_nodes = []
    padded_adjs = []
    masks = []

    for i in range(len(batch)):
        n = nodes[i].shape[0]
        # 补齐节点特征
        pad_n = torch.zeros(max_nodes - n, node_dim)
        padded_nodes.append(torch.cat([nodes[i], pad_n], dim=0))

        # 补齐邻接矩阵
        adj = adjs[i]
        pad_adj = torch.zeros(max_nodes, max_nodes)
        pad_adj[:n, :n] = adj
        padded_adjs.append(pad_adj)

        # 掩码
        mask = torch.zeros(max_nodes)
        mask[:n] = 1.0
        masks.append(mask)

    return (
        torch.stack(features),
        torch.stack(labels),
        torch.stack(padded_adjs),
        torch.stack(padded_nodes),
        torch.stack(masks),
        names,
    )


# ============================================================
# 数据加载与处理
# ============================================================

def load_data_from_directory(data_dir, label, max_samples=None):
    """从目录加载 PDB 文件并提取特征"""
    if max_samples is None:
        max_samples = CONFIG['pdb_samples_per_class']

    # 根据类别使用不同的文件模式
    file_patterns = {
        0: '**/*_protein.pdb',   # P-L: 蛋白质文件
        1: '*.pdb',               # P-P: 扁平目录的复合物文件
        2: '*.pdb',               # P-NA: 扁平目录的复合物文件
        3: '**/*_nucleic_acid.pdb',  # NA-L: 核酸文件
    }

    pattern = file_patterns.get(label, '**/*.pdb')
    pdb_files = list(Path(data_dir).glob(pattern))
    if not pdb_files:
        # 回退到通用搜索
        pdb_files = list(Path(data_dir).glob('**/*.pdb'))
    if not pdb_files:
        pdb_files = list(Path(data_dir).glob('**/*.pdb.gz'))

    # 随机采样
    np.random.seed(CONFIG['random_seed'])
    if len(pdb_files) > max_samples:
        pdb_files = np.random.choice(pdb_files, max_samples, replace=False).tolist()

    all_features = []
    all_labels = []
    all_adjs = []
    all_nodes = []
    all_names = []
    skipped = 0

    for pdb_file in pdb_files:
        try:
            features, seq, meta = extract_features(pdb_file)
            if features is None:
                skipped += 1
                continue

            # 构建图特征
            chains = parse_pdb_structure(pdb_file)
            if not chains:
                skipped += 1
                continue
            _, ca_coords, residue_names = get_sequence_and_coords(chains)
            if len(residue_names) < 10:
                skipped += 1
                continue

            # 截断
            if len(residue_names) > CONFIG['max_seq_len']:
                residue_names = residue_names[:CONFIG['max_seq_len']]
                ca_coords = ca_coords[:CONFIG['max_seq_len']]

            # 节点特征 (one-hot 氨基酸类型)
            node_feat = np.zeros((len(residue_names), CONFIG['num_aa_types']))
            for i, r in enumerate(residue_names):
                if r in AA_INDEX:
                    node_feat[i, AA_INDEX[r]] = 1.0
                else:
                    node_feat[i, 20] = 1.0  # 未知归入第20维

            # 邻接矩阵
            adj, _ = build_graph_adjacency(ca_coords)

            name = pdb_file.stem.replace('_complex', '')

            all_features.append(features)
            all_labels.append(label)
            all_adjs.append(adj)
            all_nodes.append(node_feat)
            all_names.append(name)

        except Exception as e:
            skipped += 1
            continue

    print(f"  [{CATEGORY_NAMES_CN[label]}] 加载 {len(all_features)} 个样本, 跳过 {skipped} 个")
    return all_features, all_labels, all_adjs, all_nodes, all_names


def collect_pdb_files(data_dir, label):
    """收集目录下所有匹配的 PDB 文件路径"""
    file_patterns = {
        0: '**/*_protein.pdb',
        1: '*.pdb',
        2: '*.pdb',
        3: '**/*_nucleic_acid.pdb',
    }
    pattern = file_patterns.get(label, '**/*.pdb')
    files = list(Path(data_dir).glob(pattern))
    if not files:
        files = list(Path(data_dir).glob('**/*.pdb'))
    if not files:
        files = list(Path(data_dir).glob('**/*.pdb.gz'))
    return files


def load_features_from_files(pdb_files, label):
    """从指定文件列表提取特征，返回 (features, labels, adjs, nodes, names)"""
    all_features = []
    all_labels = []
    all_adjs = []
    all_nodes = []
    all_names = []
    skipped = 0

    allow_na = (label == 3)  # NA-L 类别允许核酸链

    for pdb_file in pdb_files:
        try:
            features, seq, meta = extract_features(pdb_file)
            if features is None:
                skipped += 1
                continue

            chains = parse_pdb_structure(pdb_file, allow_na=allow_na)
            if not chains:
                skipped += 1
                continue
            _, ca_coords, residue_names = get_sequence_and_coords(chains)
            if len(residue_names) < 10:
                skipped += 1
                continue

            if len(residue_names) > CONFIG['max_seq_len']:
                residue_names = residue_names[:CONFIG['max_seq_len']]
                ca_coords = ca_coords[:CONFIG['max_seq_len']]

            node_feat = np.zeros((len(residue_names), CONFIG['num_aa_types']))
            for i, r in enumerate(residue_names):
                if r in AA_INDEX:
                    node_feat[i, AA_INDEX[r]] = 1.0
                else:
                    node_feat[i, 20] = 1.0

            adj, _ = build_graph_adjacency(ca_coords)
            name = pdb_file.stem.replace('_complex', '')

            all_features.append(features)
            all_labels.append(label)
            all_adjs.append(adj)
            all_nodes.append(node_feat)
            all_names.append(name)

        except Exception:
            skipped += 1
            continue

    return all_features, all_labels, all_adjs, all_nodes, all_names, skipped


def prepare_data():
    """准备训练/验证/测试数据 — 按类别均衡采样"""
    print("=" * 60)
    print("准备数据...")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    max_per_class = CONFIG['pdb_samples_per_class']

    all_features = []
    all_labels = []
    all_adjs = []
    all_nodes = []
    all_names = []

    np.random.seed(CONFIG['random_seed'])

    for label, data_dirs in CONFIG['data_dirs'].items():
        dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]

        # 收集该类所有目录下的 PDB 文件
        class_files = []
        for data_dir in dirs:
            path = project_root / data_dir
            if path.exists():
                files = collect_pdb_files(str(path), label)
                class_files.extend(files)
            else:
                print(f"  警告: 数据目录不存在 {path}")

        # 类别级随机采样
        if len(class_files) > max_per_class:
            class_files = np.random.choice(class_files, max_per_class, replace=False).tolist()

        print(f"  [{CATEGORY_NAMES_CN[label]}] 候选 {len(class_files)} 个文件, 开始提取特征...")

        feats, labs, adjs, nodes, names, skipped = load_features_from_files(class_files, label)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_adjs.extend(adjs)
        all_nodes.extend(nodes)
        all_names.extend(names)
        print(f"  [{CATEGORY_NAMES_CN[label]}] 加载 {len(feats)} 个样本, 跳过 {skipped} 个")

    if len(all_features) == 0:
        raise RuntimeError("没有加载到任何数据！请检查数据路径。")

    # 打乱
    indices = np.arange(len(all_features))
    np.random.seed(CONFIG['random_seed'])
    np.random.shuffle(indices)

    n = len(indices)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = train_end + int(n * CONFIG['val_ratio'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def make_dataset(idx_list):
        return ProteinDataset(
            [all_features[i] for i in idx_list],
            [all_labels[i] for i in idx_list],
            [all_adjs[i] for i in idx_list],
            [all_nodes[i] for i in idx_list],
            [all_names[i] for i in idx_list],
        )

    train_set = make_dataset(train_idx)
    val_set = make_dataset(val_idx)
    test_set = make_dataset(test_idx)

    print(f"\n总计: {n} 样本")
    print(f"  训练集: {len(train_set)}")
    print(f"  验证集: {len(val_set)}")
    print(f"  测试集: {len(test_set)}")
    for label in range(4):
        count = sum(1 for i in train_idx if all_labels[i] == label)
        print(f"  类别 [{CATEGORY_NAMES_CN[label]}]: 训练 {count}")

    return train_set, val_set, test_set, all_features[0].shape[0]


# ============================================================
# 训练与评估
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for feat, labels, adj, nodes, mask, names in loader:
        feat = feat.to(device)
        labels = labels.to(device)
        adj = adj.to(device)
        nodes = nodes.to(device)

        optimizer.zero_grad()
        outputs = model(feat, nodes, adj)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feat.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += feat.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for feat, labels, adj, nodes, mask, names in loader:
        feat = feat.to(device)
        labels = labels.to(device)
        adj = adj.to(device)
        nodes = nodes.to(device)

        outputs = model(feat, nodes, adj)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * feat.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += feat.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model():
    """主训练流程"""
    # 准备数据
    train_set, val_set, test_set, feat_dim = prepare_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 构建模型
    model = ProteinGNN(
        global_feat_dim=feat_dim,
        num_classes=4,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)

    # 类别加权 — 缓解 P-L 过多导致的偏差
    label_counts = [0, 0, 0, 0]
    for _, label, _, _, _ in train_set:
        label_counts[int(label)] += 1
    total = sum(label_counts)
    weights = []
    for c in range(4):
        count = max(label_counts[c], 1)
        weights.append(total / (4 * count))  # inverse frequency, balanced
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"\n各类样本: {label_counts}")
    print(f"类别权重: {[f'{w:.2f}' for w in weights]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            save_dir = Path(__file__).parent.parent / 'backend' / 'model'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG,
                'feat_dim': feat_dim,
                'category_names': CATEGORY_NAMES,
                'category_names_cn': CATEGORY_NAMES_CN,
            }, save_dir / 'protein_classifier.pth')
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n最佳模型: Epoch {best_epoch+1}, Val Acc: {best_val_acc:.4f}")

    # 测试
    print("\n" + "=" * 60)
    print("测试集评估...")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(
        Path(__file__).parent.parent / 'backend' / 'model' / 'protein_classifier.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n测试集准确率: {test_acc:.4f}")

    # 每类准确率
    from collections import Counter
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, t in zip(test_preds, test_labels):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    print("\n各类别准确率:")
    for label in range(4):
        if class_total[label] > 0:
            acc = class_correct[label] / class_total[label]
            print(f"  {CATEGORY_NAMES_CN[label]}: {acc:.4f} ({class_correct[label]}/{class_total[label]})")

    return model, history, test_acc


# ============================================================
# 推理接口
# ============================================================

class ProteinPredictor:
    """蛋白质预测器 - 提供 predict_protein 接口"""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            model_path = Path(__file__).parent.parent / 'backend' / 'model' / 'protein_classifier.pth'

        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', CONFIG)
        self.feat_dim = checkpoint['feat_dim']
        self.category_names = checkpoint.get('category_names', CATEGORY_NAMES)
        self.category_names_cn = checkpoint.get('category_names_cn', CATEGORY_NAMES_CN)

        self.model = ProteinGNN(
            global_feat_dim=self.feat_dim,
            num_classes=4,
            hidden_dim=self.config.get('hidden_dim', CONFIG['hidden_dim']),
            num_layers=self.config.get('num_layers', CONFIG['num_layers']),
            dropout=self.config.get('dropout', CONFIG['dropout']),
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, pdb_file_path):
        """预测单个 PDB 文件的蛋白质类别"""
        # 提取特征
        features, seq, meta = extract_features(pdb_file_path)
        if features is None:
            raise ValueError(f"无法解析 PDB 文件: {pdb_file_path}")

        # 提取图特征
        chains = parse_pdb_structure(pdb_file_path, allow_na=True)
        _, ca_coords, residue_names = get_sequence_and_coords(chains)

        if len(residue_names) > CONFIG['max_seq_len']:
            residue_names = residue_names[:CONFIG['max_seq_len']]
            ca_coords = ca_coords[:CONFIG['max_seq_len']]

        # 节点特征
        node_feat = np.zeros((len(residue_names), CONFIG['num_aa_types']))
        for i, r in enumerate(residue_names):
            if r in AA_INDEX:
                node_feat[i, AA_INDEX[r]] = 1.0
            else:
                node_feat[i, 20] = 1.0

        adj, _ = build_graph_adjacency(ca_coords)

        # 转换成张量
        feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        node_tensor = torch.FloatTensor(node_feat).unsqueeze(0).to(self.device)
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(feat_tensor, node_tensor, adj_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # 生成蛋白质名字
        pdb_name = Path(pdb_file_path).stem.replace('_complex', '')
        protein_name = self._generate_protein_name(pdb_name, pred_class, seq)

        result = {
            'name': protein_name,
            'category': self.category_names[pred_class],
            'category_cn': self.category_names_cn[pred_class],
            'category_id': pred_class,
            'confidence': confidence,
            'pdb_id': pdb_name,
            'sequence_length': len(residue_names),
            'all_probabilities': {
                self.category_names_cn[i]: probs[0, i].item()
                for i in range(4)
            }
        }

        return result

    def _generate_protein_name(self, pdb_name, category, sequence):
        """基于 PDB ID 和类别生成蛋白质名称"""
        category_prefix = {
            0: '结合蛋白',
            1: '复合物蛋白',
            2: '核酸结合蛋白',
            3: '核酸',
        }
        prefix = category_prefix.get(category, '蛋白质')
        return f"{prefix}_{pdb_name}"


# 全局预测器实例(延迟加载)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        model_path = Path(__file__).parent.parent / 'backend' / 'model' / 'protein_classifier.pth'
        _predictor = ProteinPredictor(str(model_path))
    return _predictor


def predict_protein(pdb_file_path):
    """
    对外接口函数：预测蛋白质名字和分类

    Args:
        pdb_file_path: PDB 文件路径

    Returns:
        dict: {"name": ..., "category": ..., "category_cn": ..., "confidence": ...}
    """
    predictor = get_predictor()
    return predictor.predict(pdb_file_path)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("蛋白质判别模型训练系统")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # 检查数据
    data_available = False
    for d in CONFIG['data_dirs'].values():
        dirs = d if isinstance(d, list) else [d]
        if any((project_root / sub).exists() for sub in dirs):
            data_available = True
            break

    if not data_available:
        print("\n未找到训练数据。请先将 PDB 文件放入 data/ 目录。")
        print("执行以下命令解压数据:")
        print("  cd protein_classifier_project")
        print("  tar -xzf ../P-L.tar.gz -C ../data/")
        print("  tar -xzf ../P-P.tar.gz -C ../data/")
        print("  tar -xzf ../P-NA.tar.gz -C ../data/")
        print("  tar -xzf ../NA-L.tar.gz -C ../data/")
        print("\n或者运行 prepare_data.py 准备示例数据:")
        print("  python prepare_data.py --p-l ../P-L.tar.gz --p-p ../P-P.tar.gz ...")
        sys.exit(1)

    # 训练
    model, history, test_acc = train_model()

    # 示例预测
    print("\n" + "=" * 60)
    print("测试 predict_protein 接口...")
    print("=" * 60)

    # 找一个测试样本
    all_pdbs = []
    for label, data_dirs in CONFIG['data_dirs'].items():
        dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        for data_dir in dirs:
            path = project_root / data_dir
            if path.exists():
                all_pdbs.extend(list(path.glob('**/*.pdb'))[:3])

    if all_pdbs:
        for pdb_file in all_pdbs[:3]:
            try:
                result = predict_protein(str(pdb_file))
                print(f"\n文件: {pdb_file.name}")
                print(f"  名字: {result['name']}")
                print(f"  分类: {result['category_cn']}")
                print(f"  置信度: {result['confidence']:.4f}")
            except Exception as e:
                print(f"  预测失败: {e}")

    print("\n训练完成！")
