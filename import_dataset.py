"""
将数据集中的 PDB 文件批量导入数据库
根据文件所在目录确定分类 (ground truth)
支持增量导入 — 跳过已存在的 pdb_id
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.database.init_db import init_database, insert_protein, get_connection

# 数据集路径 (相对于项目根目录)
DATA_ROOT = Path(__file__).parent.parent / 'data'

CATEGORY_MAP = {
    # P-L (酶/受体) — 来自多个 PDBbind 版本和 PLANET 数据集
    'P-L':                          ('酶/受体 (Protein-Ligand)',                     '酶/受体'),
    'v2016':                        ('酶/受体 (Protein-Ligand)',                     '酶/受体'),
    'v2017':                        ('酶/受体 (Protein-Ligand)',                     '酶/受体'),
    'v2018':                        ('酶/受体 (Protein-Ligand)',                     '酶/受体'),
    'v2019':                        ('酶/受体 (Protein-Ligand)',                     '酶/受体'),
    'PLANET_dataset/PDBbind2020-PLANET': ('酶/受体 (Protein-Ligand)',                '酶/受体'),
    # P-P
    'P-P':  ('蛋白质-蛋白质复合物 (Protein-Protein)',          '蛋白质-蛋白质复合物'),
    # P-NA
    'P-NA': ('蛋白质-核酸复合物 (Protein-Nucleic Acid)',        '蛋白质-核酸复合物'),
    # NA-L
    'NA-L': ('核酸-配体复合物 (Nucleic Acid-Ligand)',           '核酸-配体复合物'),
}

NAME_PREFIX = {
    '酶/受体':             '结合蛋白',
    '蛋白质-蛋白质复合物':   '复合物蛋白',
    '蛋白质-核酸复合物':    '核酸结合蛋白',
    '核酸-配体复合物':      '核酸',
}


def collect_entries():
    """扫描所有数据集目录，返回去重后的 (pdb_id, pdb_file, category_en, category_cn) 列表"""
    seen = {}  # pdb_id -> (pdb_file, category_en, category_cn)

    for category_key, (category_en, category_cn) in CATEGORY_MAP.items():
        cat_dir = DATA_ROOT / category_key
        if not cat_dir.exists():
            print(f"  [SKIP] 目录不存在: {cat_dir}")
            continue

        found = 0
        for pdb_file in cat_dir.glob('**/*.pdb'):
            stem = pdb_file.stem
            for suffix in ['_complex', '_protein', '_nucleic_acid']:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            pdb_id = stem

            if pdb_id not in seen:
                seen[pdb_id] = (pdb_file, category_en, category_cn)
            found += 1

        print(f"  [{category_key}] 扫描 {found} 个 PDB 文件")

    return seen


def main():
    init_database()
    print()

    # 收集所有待导入条目 (已按 pdb_id 去重)
    entries = collect_entries()
    print(f"\n去重后共 {len(entries)} 个唯一 PDB ID\n")

    # 获取数据库中已有的 pdb_id
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT pdb_id FROM proteins')
    existing_ids = {row[0] for row in cursor.fetchall()}
    print(f"数据库中已有 {len(existing_ids)} 条记录\n")

    # 筛选出新条目
    new_entries = []
    for pdb_id, (pdb_file, category_en, category_cn) in entries.items():
        if pdb_id not in existing_ids:
            new_entries.append((pdb_id, pdb_file, category_en, category_cn))

    print(f"待新增: {len(new_entries)} 条")

    if not new_entries:
        print("没有新记录需要导入。")
        conn.close()
        return

    # 批量插入
    BATCH_SIZE = 500
    inserted = 0
    batch = []

    for pdb_id, pdb_file, category_en, category_cn in new_entries:
        name = f"{NAME_PREFIX.get(category_cn, '蛋白质')}_{pdb_id}"
        batch.append((name, category_en, category_cn, str(pdb_file),
                      pdb_id, None, None,  # confidence=NULL 表示已知分类(非预测)
                      f"来自数据集, PDB ID: {pdb_id}"))

        if len(batch) >= BATCH_SIZE:
            cursor.executemany('''
                INSERT INTO proteins (name, category, category_cn, pdb_file_path,
                                      pdb_id, confidence, sequence_length, additional_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
            conn.commit()
            inserted += len(batch)
            print(f"  已导入 {inserted} 条...")
            batch = []

    # 剩余
    if batch:
        cursor.executemany('''
            INSERT INTO proteins (name, category, category_cn, pdb_file_path,
                                  pdb_id, confidence, sequence_length, additional_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch)
        conn.commit()
        inserted += len(batch)

    conn.close()
    print(f"\n完成! 新增 {inserted} 条记录")


if __name__ == '__main__':
    main()
