"""
从 RCSB PDB API 批量获取蛋白质真实名称并更新数据库 (并发版本)
"""

import sys
import re
import time
import sqlite3
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, str(Path(__file__).parent))

from backend.database.init_db import DB_PATH

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'HSE': 'H', 'HSD': 'H', 'HSP': 'H', 'HIE': 'H',
}

AA_NAME_CN = {
    'A': '丙氨酸', 'C': '半胱氨酸', 'D': '天冬氨酸', 'E': '谷氨酸', 'F': '苯丙氨酸',
    'G': '甘氨酸', 'H': '组氨酸', 'I': '异亮氨酸', 'K': '赖氨酸', 'L': '亮氨酸',
    'M': '甲硫氨酸', 'N': '天冬酰胺', 'P': '脯氨酸', 'Q': '谷氨酰胺', 'R': '精氨酸',
    'S': '丝氨酸', 'T': '苏氨酸', 'V': '缬氨酸', 'W': '色氨酸', 'Y': '酪氨酸',
}

CATEGORY_INFO_CN = {
    '酶/受体': '该蛋白质属于酶或受体类，在生物体内参与催化化学反应或信号转导过程，是药物开发的重要靶点。',
    '蛋白质-蛋白质复合物': '该蛋白质形成蛋白质-蛋白质复合物，参与蛋白间相互作用、信号通路传导及细胞调控等关键生物学过程。',
    '蛋白质-核酸复合物': '该蛋白质与DNA或RNA形成复合物，参与基因调控、DNA复制、转录或翻译等遗传信息处理过程。',
    '核酸-配体复合物': '该结构主要为核酸分子与小分子配体的复合物，在基因表达调控和药物设计中具有重要意义。',
}

TERM_TRANSLATIONS_EN2CN = {
    'complexed with': '与...复合',
    'crystal structure of': '晶体结构',
    'solution structure of': '溶液结构',
    'nmr structure of': '核磁共振结构',
    'complex': '复合物', 'domain': '结构域', 'inhibitor': '抑制剂',
    'receptor': '受体', 'enzyme': '酶', 'kinase': '激酶',
    'protease': '蛋白酶', 'transferase': '转移酶', 'hydrolase': '水解酶',
    'oxidoreductase': '氧化还原酶', 'lyase': '裂解酶', 'isomerase': '异构酶',
    'ligase': '连接酶', 'dehydrogenase': '脱氢酶',
    'binding protein': '结合蛋白', 'transcription': '转录', 'factor': '因子',
    'antibody': '抗体', 'antigen': '抗原', 'mutant': '突变体',
    'wild-type': '野生型', 'peptide': '肽', 'substrate': '底物',
    'ligand': '配体', 'analog': '类似物', 'DNA': 'DNA', 'RNA': 'RNA',
    'GDP': 'GDP', 'GTP': 'GTP', 'ATP': 'ATP',
    'NADH': 'NADH', 'NADPH': 'NADPH',
    'human': '人', 'mouse': '小鼠', 'rat': '大鼠', 'bovine': '牛',
    'synthetic': '合成', 'construct': '构建体',
    'in complex with': '与...形成复合物',
    'bound to': '与...结合',
}

NAME_PREFIX = {
    '酶/受体': '结合蛋白',
    '蛋白质-蛋白质复合物': '复合物蛋白',
    '蛋白质-核酸复合物': '核酸结合蛋白',
    '核酸-配体复合物': '核酸',
}


def parse_pdb_sequence(pdb_path):
    chains = {}
    seq_count = Counter()
    atom_count = 0
    ca_atom_count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_count += 1
                    if line[12:16].strip() == 'CA':
                        ca_atom_count += 1
                    chain = line[21]
                    res_name = line[17:20].strip()
                    aa = AA_3TO1.get(res_name, 'X')
                    seq_count[aa] += 1
                elif line.startswith('SEQRES'):
                    chain = line[11]
                    if chain not in chains:
                        chains[chain] = []
                    residues = line[19:].split()
                    for r in residues:
                        aa = AA_3TO1.get(r, 'X')
                        seq_count[aa] += 1
    except Exception:
        pass
    total_residues = sum(seq_count.values())
    top_aa = [AA_NAME_CN.get(a, a) for a, _ in seq_count.most_common(5)]
    return {
        'chain_count': len(set(chains.keys())) or 1,
        'atom_count': atom_count,
        'ca_count': ca_atom_count,
        'residue_count': total_residues,
        'top_amino_acids': top_aa,
    }


def fetch_pdb_metadata(pdb_id, session):
    url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}'
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            title = data.get('struct', {}).get('title', '')
            keywords = data.get('struct_keywords', {}).get('pdbx_keywords', '')
            return {'title': title, 'keywords': keywords}
    except Exception:
        pass
    return None


def translate_title(title):
    if not title:
        return ''
    result = title
    for en, cn in sorted(TERM_TRANSLATIONS_EN2CN.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(en), re.IGNORECASE)
        result = pattern.sub(cn, result)
    return result


def generate_description(pdb_id, category_cn, meta, structure_info):
    parts = []
    cat_desc = CATEGORY_INFO_CN.get(category_cn, '')
    if cat_desc:
        parts.append(cat_desc)
    if structure_info:
        si = structure_info
        parts.append(
            f"该结构包含{si['chain_count']}条多肽链，"
            f"共{si['residue_count']}个氨基酸残基，"
            f"{si['ca_count']}个Cα原子。"
        )
        if si['top_amino_acids']:
            parts.append(f"主要氨基酸组成为：{'、'.join(si['top_amino_acids'][:5])}。")
    if meta and meta.get('title'):
        cn_title = translate_title(meta['title'])
        parts.append(f"蛋白质名称：{meta['title']}。")
        if cn_title != meta['title']:
            parts.append(f"功能描述：{cn_title}。")
    parts.append(f"PDB标识符：{pdb_id}。")
    return ''.join(parts)


def process_one(row, session):
    """处理单条记录，返回 (pdb_id, name, additional_info)"""
    pdb_id = row['pdb_id']
    category_cn = row['category_cn']
    pdb_path = row['pdb_file_path']

    structure_info = None
    if pdb_path and Path(pdb_path).exists():
        structure_info = parse_pdb_sequence(pdb_path)

    meta = fetch_pdb_metadata(pdb_id, session)

    if meta and meta.get('title'):
        name = meta['title']
        if len(name) > 150:
            name = name[:147] + '...'
    else:
        prefix = NAME_PREFIX.get(category_cn, '蛋白质')
        name = f"{prefix}_{pdb_id}"

    description = generate_description(pdb_id, category_cn, meta, structure_info)
    return pdb_id, name, description, meta is not None and meta.get('title') != ''


def main():
    log_file = Path(__file__).parent / 'enrich_log.txt'
    log_file.write_text('', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log("=" * 60)
    log("蛋白质数据富集 (并发版) - 从RCSB PDB获取名称")
    log("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, pdb_id, category_cn, pdb_file_path FROM proteins')
    all_rows = cursor.fetchall()
    conn.close()

    total = len(all_rows)
    log(f"\n共 {total} 条记录，使用10线程并发请求\n")

    api_ok = 0
    api_fail = 0
    completed = 0

    session = requests.Session()
    session.headers.update({'User-Agent': 'ProteinClassifier/1.0'})

    write_conn = sqlite3.connect(str(DB_PATH))
    write_cur = write_conn.cursor()

    batch_updates = []
    BATCH_SIZE = 100

    def flush_batch():
        nonlocal batch_updates
        if not batch_updates:
            return
        write_cur.executemany(
            '''UPDATE proteins SET name=?, additional_info=?, updated_at=CURRENT_TIMESTAMP
               WHERE pdb_id=?''',
            batch_updates
        )
        write_conn.commit()
        batch_updates = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_one, row, session): row for row in all_rows}

        for future in as_completed(futures):
            try:
                pdb_id, name, desc, got_name = future.result()
            except Exception as e:
                continue

            if got_name:
                api_ok += 1
            else:
                api_fail += 1

            batch_updates.append((name, desc, pdb_id))
            completed += 1

            if len(batch_updates) >= BATCH_SIZE:
                flush_batch()
                log(f"  进度: {completed}/{total} (API成功: {api_ok}, 失败: {api_fail})")

    flush_batch()
    write_conn.close()

    log(f"\n{'=' * 60}")
    log(f"完成! 共更新 {completed} 条记录")
    log(f"API 成功获取名称: {api_ok}")
    log(f"API 未能获取: {api_fail}")
    log(f"{'=' * 60}")


if __name__ == '__main__':
    main()
