"""
工具函数：中文翻译、文件处理、数据库操作
"""

# 蛋白质类别中英文映射
CATEGORY_TRANSLATIONS = {
    '酶/受体 (Protein-Ligand)': '酶/受体',
    '蛋白质-蛋白质复合物 (Protein-Protein)': '蛋白质-蛋白质复合物',
    '蛋白质-核酸复合物 (Protein-Nucleic Acid)': '蛋白质-核酸复合物',
    '核酸-配体复合物 (Nucleic Acid-Ligand)': '核酸-配体复合物',
    '酶/受体': '酶/受体 (Protein-Ligand)',
    '蛋白质-蛋白质复合物': '蛋白质-蛋白质复合物 (Protein-Protein)',
    '蛋白质-核酸复合物': '蛋白质-核酸复合物 (Protein-Nucleic Acid)',
    '核酸-配体复合物': '核酸-配体复合物 (Nucleic Acid-Ligand)',
}

# 蛋白质相关术语中英文对照表
TERM_TRANSLATIONS = {
    # 氨基酸
    'Alanine': '丙氨酸', 'Cysteine': '半胱氨酸', 'Aspartate': '天冬氨酸',
    'Glutamate': '谷氨酸', 'Phenylalanine': '苯丙氨酸', 'Glycine': '甘氨酸',
    'Histidine': '组氨酸', 'Isoleucine': '异亮氨酸', 'Lysine': '赖氨酸',
    'Leucine': '亮氨酸', 'Methionine': '甲硫氨酸', 'Asparagine': '天冬酰胺',
    'Proline': '脯氨酸', 'Glutamine': '谷氨酰胺', 'Arginine': '精氨酸',
    'Serine': '丝氨酸', 'Threonine': '苏氨酸', 'Valine': '缬氨酸',
    'Tryptophan': '色氨酸', 'Tyrosine': '酪氨酸',
    # 蛋白质类型
    'enzyme': '酶', 'receptor': '受体', 'kinase': '激酶',
    'protease': '蛋白酶', 'inhibitor': '抑制剂', 'antibody': '抗体',
    'transporter': '转运蛋白', 'channel': '通道蛋白',
    'transcription factor': '转录因子', 'hormone': '激素',
    'cytochrome': '细胞色素', 'hemoglobin': '血红蛋白',
    'insulin': '胰岛素', 'albumin': '白蛋白',
    'collagen': '胶原蛋白', 'actin': '肌动蛋白',
    'myosin': '肌球蛋白', 'tubulin': '微管蛋白',
    # 复合物类型
    'complex': '复合物', 'protein': '蛋白质',
    'ligand': '配体', 'nucleic acid': '核酸',
    'DNA': 'DNA', 'RNA': 'RNA',
    'binding protein': '结合蛋白', 'complex protein': '复合物蛋白',
    'nucleic acid binding protein': '核酸结合蛋白',
    # 结构相关
    'alpha helix': 'α螺旋', 'beta sheet': 'β折叠',
    'active site': '活性位点', 'binding site': '结合位点',
    'domain': '结构域', 'subunit': '亚基',
    'chain': '链', 'residue': '残基',
}

# 蛋白质名称中英文翻译
PROTEIN_NAME_TRANSLATIONS = {
    '结合蛋白': 'Binding Protein',
    '复合物蛋白': 'Complex Protein',
    '核酸结合蛋白': 'Nucleic Acid Binding Protein',
    '核酸': 'Nucleic Acid',
}


def translate_category(category_text):
    """翻译蛋白质分类为中文"""
    return CATEGORY_TRANSLATIONS.get(category_text, category_text)


def translate_protein_name(name):
    """翻译蛋白质名称为中文（简单规则翻译）"""
    # 替换常见英文术语为中文
    translated = name
    for en, cn in TERM_TRANSLATIONS.items():
        # 大小写不敏感替换
        pattern = en
        if pattern.lower() in translated.lower():
            # 使用正则替换保持大小写
            import re
            translated = re.sub(pattern, cn, translated, flags=re.IGNORECASE)

    # 处理前缀
    for cn, en in PROTEIN_NAME_TRANSLATIONS.items():
        if name.startswith(cn):
            return name  # 已经是中文

    return translated


def translate_to_chinese(text):
    """通用翻译函数：将英文蛋白质相关文本翻译为中文"""
    if not text:
        return text

    result = text

    # 先处理类别名
    if result in CATEGORY_TRANSLATIONS:
        return CATEGORY_TRANSLATIONS[result]

    # 处理蛋白质名 (提取前缀)
    for cn, en in PROTEIN_NAME_TRANSLATIONS.items():
        if result.startswith(cn):
            # 已经是中文，直接返回
            return result

    # 替换已知术语
    for en, cn in sorted(TERM_TRANSLATIONS.items(),
                          key=lambda x: len(x[0]), reverse=True):
        import re
        result = re.sub(en, cn, result, flags=re.IGNORECASE)

    return result


def allowed_file(filename, allowed_extensions=None):
    """检查文件扩展名是否允许"""
    if allowed_extensions is None:
        allowed_extensions = {'pdb', 'pdb.gz', 'ent', 'cif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions or \
           filename.endswith('.pdb.gz')


def get_file_extension(filename):
    """获取文件扩展名"""
    if filename.endswith('.pdb.gz'):
        return 'pdb.gz'
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def generate_protein_info(result):
    """根据预测结果生成蛋白质描述信息"""
    category_id = result.get('category_id', 0)
    category_cn = result.get('category_cn', '未知')
    pdb_id = result.get('pdb_id', '未知')
    seq_len = result.get('sequence_length', 0)
    confidence = result.get('confidence', 0)

    descriptions = {
        0: f"该蛋白质属于酶/受体类，可能参与生物催化或信号转导过程。PDB ID: {pdb_id}，"
           f"蛋白质链长度约 {seq_len} 个氨基酸残基，预测置信度 {confidence:.1%}。",
        1: f"该蛋白质形成蛋白质-蛋白质复合物，可能参与蛋白间相互作用和信号通路。PDB ID: {pdb_id}，"
           f"蛋白质链长度约 {seq_len} 个氨基酸残基，预测置信度 {confidence:.1%}。",
        2: f"该蛋白质与核酸(DNA/RNA)形成复合物，可能参与基因调控、复制或转录过程。PDB ID: {pdb_id}，"
           f"蛋白质链长度约 {seq_len} 个氨基酸残基，预测置信度 {confidence:.1%}。",
        3: f"该结构主要为核酸分子与配体的复合物。PDB ID: {pdb_id}，"
           f"链长度约 {seq_len} 个残基，预测置信度 {confidence:.1%}。",
    }

    return descriptions.get(category_id, f"蛋白质 {pdb_id}，分类为 {category_cn}。")
