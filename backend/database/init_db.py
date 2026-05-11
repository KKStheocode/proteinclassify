"""
数据库初始化与操作模块
使用 SQLite 存储蛋白质识别结果
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path


DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / 'proteins.db'


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """初始化数据库 - 创建表结构"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS proteins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            category_cn TEXT NOT NULL,
            pdb_file_path TEXT,
            pdb_id TEXT,
            confidence REAL,
            sequence_length INTEGER,
            additional_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建索引加速查询
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_proteins_name
        ON proteins(name)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_proteins_category
        ON proteins(category)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_proteins_category_cn
        ON proteins(category_cn)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_proteins_pdb_id
        ON proteins(pdb_id)
    ''')

    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {DB_PATH}")


def insert_protein(name, category, category_cn, pdb_file_path=None,
                   pdb_id=None, confidence=None, sequence_length=None,
                   additional_info=None):
    """插入蛋白质记录"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO proteins (name, category, category_cn, pdb_file_path,
                              pdb_id, confidence, sequence_length, additional_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, category, category_cn, pdb_file_path, pdb_id,
          confidence, sequence_length, additional_info))

    protein_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return protein_id


def update_protein(protein_id, name=None, category=None, category_cn=None,
                   confidence=None, additional_info=None):
    """更新蛋白质记录"""
    conn = get_connection()
    cursor = conn.cursor()

    fields = []
    values = []

    if name is not None:
        fields.append('name = ?')
        values.append(name)
    if category is not None:
        fields.append('category = ?')
        values.append(category)
    if category_cn is not None:
        fields.append('category_cn = ?')
        values.append(category_cn)
    if confidence is not None:
        fields.append('confidence = ?')
        values.append(confidence)
    if additional_info is not None:
        fields.append('additional_info = ?')
        values.append(additional_info)

    if fields:
        fields.append('updated_at = CURRENT_TIMESTAMP')
        values.append(protein_id)
        cursor.execute(f'''
            UPDATE proteins SET {', '.join(fields)}
            WHERE id = ?
        ''', values)

    conn.commit()
    conn.close()


def search_proteins(query=None, category_cn=None, limit=100):
    """搜索蛋白质 - 按名字或分类"""
    conn = get_connection()
    cursor = conn.cursor()

    conditions = []
    params = []

    if query:
        conditions.append('(name LIKE ? OR pdb_id LIKE ? OR additional_info LIKE ?)')
        search_term = f'%{query}%'
        params.extend([search_term, search_term, search_term])

    if category_cn:
        conditions.append('category_cn = ?')
        params.append(category_cn)

    where_clause = ' AND '.join(conditions) if conditions else '1=1'

    cursor.execute(f'''
        SELECT * FROM proteins
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
    ''', params + [limit])

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_protein_by_id(protein_id):
    """通过 ID 获取蛋白质"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM proteins WHERE id = ?', (protein_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_protein_by_pdb_id(pdb_id):
    """通过 PDB ID 获取蛋白质"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM proteins WHERE pdb_id = ?', (pdb_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_proteins(limit=100, offset=0):
    """获取所有蛋白质记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM proteins
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_statistics():
    """获取数据库统计信息"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total FROM proteins')
    total = cursor.fetchone()['total']

    cursor.execute('''
        SELECT category_cn, COUNT(*) as count
        FROM proteins
        GROUP BY category_cn
        ORDER BY count DESC
    ''')
    by_category = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {'total': total, 'by_category': by_category}


def save_prediction_result(result, pdb_file_path=None, additional_info=None):
    """
    保存预测结果到数据库
    如果已存在相同 PDB ID 的记录则更新，否则插入新记录
    """
    pdb_id = result.get('pdb_id', '')
    existing = get_protein_by_pdb_id(pdb_id)

    if existing:
        update_protein(
            existing['id'],
            name=result.get('name'),
            category=result.get('category'),
            category_cn=result.get('category_cn'),
            confidence=result.get('confidence'),
            additional_info=additional_info,
        )
        return existing['id']
    else:
        return insert_protein(
            name=result.get('name', ''),
            category=result.get('category', ''),
            category_cn=result.get('category_cn', ''),
            pdb_file_path=pdb_file_path,
            pdb_id=pdb_id,
            confidence=result.get('confidence'),
            sequence_length=result.get('sequence_length'),
            additional_info=additional_info,
        )


# 初始化
if __name__ == '__main__':
    init_database()
    print("数据库已就绪。")
