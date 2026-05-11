"""
数据准备工具
从 PDBbind 压缩包中提取样本 PDB 文件用于训练和测试
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path
import random

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


def extract_samples(archive_path, output_dir, pattern, max_count=100, flat=False):
    """
    从 tar.gz 压缩包中提取样本文件

    Args:
        archive_path: tar.gz 文件路径
        output_dir: 输出目录
        pattern: 文件名匹配模式 (如 '_protein.pdb', '_complex.pdb')
        max_count: 最多提取数量
        flat: 是否为扁平目录结构 (P-P, P-NA 是扁平的)
    """
    print(f"正在处理: {archive_path}")
    print(f"  输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(archive_path, 'r:gz') as tar:
        # 获取匹配的文件列表
        if flat:
            members = [m for m in tar.getmembers()
                      if m.name.endswith(pattern) and m.isfile()]
        else:
            members = [m for m in tar.getmembers()
                      if pattern in m.name and m.isfile()]

        print(f"  找到 {len(members)} 个匹配文件")

        # 随机采样
        if len(members) > max_count:
            random.seed(42)
            members = random.sample(members, max_count)

        # 提取
        count = 0
        for member in members:
            tar.extract(member, path=output_dir)
            count += 1
            if count % 50 == 0:
                print(f"  已提取 {count}/{len(members)}...")

        print(f"  完成: 提取了 {count} 个文件")


def main():
    parser = argparse.ArgumentParser(description='准备 PDBbind 训练数据')
    parser.add_argument('--p-l', type=str, default='P-L.tar.gz',
                       help='P-L 压缩包路径')
    parser.add_argument('--p-p', type=str, default='P-P.tar.gz',
                       help='P-P 压缩包路径')
    parser.add_argument('--p-na', type=str, default='P-NA.tar.gz',
                       help='P-NA 压缩包路径')
    parser.add_argument('--na-l', type=str, default='NA-L.tar.gz',
                       help='NA-L 压缩包路径')
    parser.add_argument('--samples', type=int, default=200,
                       help='每类提取样本数')
    parser.add_argument('--output', type=str, default='data',
                       help='输出基础目录')
    args = parser.parse_args()

    output_base = ROOT_DIR / args.output

    tasks = [
        (args.p_l, output_base / 'P-L', '_protein.pdb', args.samples, False),
        (args.p_p, output_base / 'P-P', '_complex.pdb', args.samples, True),
        (args.p_na, output_base / 'P-NA', '_complex.pdb', args.samples, True),
        (args.na_l, output_base / 'NA-L', '_nucleic_acid.pdb', min(args.samples, 50), False),
    ]

    for archive, out_dir, pattern, count, flat in tasks:
        archive_path = ROOT_DIR / archive
        if archive_path.exists():
            extract_samples(str(archive_path), str(out_dir), pattern, count, flat)
        else:
            print(f"跳过 (文件不存在): {archive_path}")

    print("\n数据准备完成！")
    print("现在可以运行 train.py 开始训练模型。")


if __name__ == '__main__':
    main()
