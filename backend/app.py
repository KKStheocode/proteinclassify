"""
蛋白质判别系统 - FastAPI 后端服务
提供 /predict 和 /search API 接口
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils import (
    translate_category, translate_to_chinese,
    allowed_file, generate_protein_info,
)
from backend.database.init_db import (
    init_database, save_prediction_result,
    search_proteins, get_all_proteins, get_statistics,
    get_protein_by_id, get_protein_by_pdb_id,
)

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(
    title="蛋白质判别系统 API",
    description="基于深度学习的蛋白质结构分类与识别系统",
    version="1.0.0",
)

# CORS 配置 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传文件存储目录
UPLOAD_DIR = Path(__file__).parent.parent / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 初始化数据库
init_database()

# 延迟加载预测器
_predictor = None


def get_predictor():
    """获取预测器实例 (延迟加载)"""
    global _predictor
    if _predictor is None:
        try:
            from model_training.train import ProteinPredictor
            model_path = Path(__file__).parent / 'model' / 'protein_classifier.pth'
            _predictor = ProteinPredictor(str(model_path))
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="模型尚未训练，请先运行训练脚本。"
            )
    return _predictor


# ============================================================
# API 接口
# ============================================================

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "蛋白质判别系统",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST) - 上传 PDB 文件进行预测",
            "search": "/search (GET) - 搜索蛋白质数据库",
            "proteins": "/proteins (GET) - 获取蛋白质列表",
            "stats": "/stats (GET) - 获取统计信息",
        }
    }


@app.post("/predict")
async def predict_protein_endpoint(file: UploadFile = File(...)):
    """
    上传 PDB 文件，返回蛋白质名字和分类

    Args:
        file: PDB 格式的蛋白质结构文件

    Returns:
        JSON: 包含蛋白质名字、分类、置信度等信息
    """
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="请提供文件名。")

    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式: .pdb, .pdb.gz, .ent, .cif"
        )

    # 保存上传文件
    file_ext = Path(file.filename).suffix
    if file.filename.endswith('.pdb.gz'):
        file_ext = '.pdb.gz'

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = UPLOAD_DIR / unique_name

    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 执行预测
    try:
        predictor = get_predictor()
        result = predictor.predict(str(file_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

    # 使用原始文件名修正 PDB ID 和名称
    original_stem = Path(file.filename).stem.replace('_complex', '').replace('_protein', '')
    result['pdb_id'] = original_stem
    # 重新生成名称
    from backend.utils import CATEGORY_TRANSLATIONS
    category_cn = CATEGORY_TRANSLATIONS.get(result.get('category', ''), result.get('category', ''))
    category_prefix = {
        '酶/受体': '结合蛋白',
        '蛋白质-蛋白质复合物': '复合物蛋白',
        '蛋白质-核酸复合物': '核酸结合蛋白',
        '核酸-配体复合物': '核酸',
    }
    prefix = category_prefix.get(category_cn, '蛋白质')
    result['name'] = f"{prefix}_{original_stem}"

    # 生成中文描述信息
    additional_info = generate_protein_info(result)

    # 保存到数据库
    try:
        protein_id = save_prediction_result(
            result,
            pdb_file_path=str(file_path),
            additional_info=additional_info,
        )
        result['id'] = protein_id
        result['db_id'] = protein_id
        result['pdb_file_path'] = str(file_path)
        result['additional_info'] = additional_info
    except Exception as e:
        result['db_error'] = str(e)

    # 确保返回中文分类
    result['category_cn_display'] = translate_category(result.get('category', ''))

    return JSONResponse(content=result)


@app.get("/search")
async def search_endpoint(
    q: str = Query(None, description="搜索关键词 (名字或 PDB ID)"),
    category: str = Query(None, description="分类过滤 (中文或英文)"),
    limit: int = Query(100, ge=1, le=1000, description="返回结果数量上限"),
):
    """
    搜索蛋白质数据库

    Args:
        q: 搜索关键词
        category: 分类过滤
        limit: 返回结果上限

    Returns:
        JSON: 匹配的蛋白质列表
    """
    try:
        # 将可能的英文分类翻译为中文, 用于搜索 category_cn 列
        category_cn = None
        if category:
            from backend.utils import CATEGORY_TRANSLATIONS
            cn_keys = {'酶/受体', '蛋白质-蛋白质复合物', '蛋白质-核酸复合物', '核酸-配体复合物'}
            if category in cn_keys:
                category_cn = category
            else:
                category_cn = CATEGORY_TRANSLATIONS.get(category, category)

        results = search_proteins(query=q, category_cn=category_cn, limit=limit)
        return JSONResponse(content={
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/proteins")
async def list_proteins(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """获取蛋白质列表"""
    try:
        results = get_all_proteins(limit=limit, offset=offset)
        stats = get_statistics()
        return JSONResponse(content={
            "total": stats['total'],
            "count": len(results),
            "offset": offset,
            "results": results,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/protein/{protein_id}")
async def get_protein(protein_id: int):
    """获取单个蛋白质详情"""
    result = get_protein_by_id(protein_id)
    if not result:
        raise HTTPException(status_code=404, detail="蛋白质记录不存在。")
    return JSONResponse(content=result)


@app.get("/stats")
async def get_stats():
    """获取数据库统计信息"""
    try:
        stats = get_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    model_loaded = _predictor is not None
    model_file = Path(__file__).parent / 'model' / 'protein_classifier.pth'
    return {
        "status": "ok",
        "model_loaded": model_loaded or model_file.exists(),
        "model_file_exists": model_file.exists(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/protein/{protein_id}/pdb")
async def get_protein_pdb(protein_id: int):
    """
    返回蛋白质的原始 PDB 文件内容, 用于 3D 可视化。

    Returns:
        JSON: {protein_id, pdb_id, name, pdb_content, file_size}
    """
    protein = get_protein_by_id(protein_id)
    if not protein:
        raise HTTPException(status_code=404, detail="蛋白质记录不存在。")

    pdb_path = protein.get("pdb_file_path")
    if not pdb_path:
        raise HTTPException(
            status_code=404,
            detail="此蛋白质没有可用的 PDB 结构文件。"
        )

    path = Path(pdb_path)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDB 结构文件未找到，可能已被移动或删除。"
        )

    file_size = path.stat().st_size
    if file_size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"PDB 文件过大 ({file_size / 1024 / 1024:.1f} MB)，最大允许 100 MB。"
        )

    try:
        pdb_content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 PDB 文件失败: {str(e)}")

    return JSONResponse(content={
        "protein_id": protein_id,
        "pdb_id": protein.get("pdb_id"),
        "name": protein.get("name"),
        "pdb_content": pdb_content,
        "file_size": file_size,
    })


# ============================================================
# 静态文件服务 (生产环境推荐使用 Nginx)
# ============================================================

frontend_dir = Path(__file__).parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ============================================================
# 启动入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("蛋白质判别系统 - 后端服务启动")
    print("=" * 60)

    # 检查模型
    model_path = Path(__file__).parent / 'model' / 'protein_classifier.pth'
    if model_path.exists():
        print(f"[OK] 模型文件已找到: {model_path}")
    else:
        print(f"[MISSING] 模型文件未找到: {model_path}")
        print("  请先运行 model_training/train.py 训练模型")

    # 检查数据库
    db_path = Path(__file__).parent / 'database' / 'proteins.db'
    if db_path.exists():
        print(f"[OK] 数据库已就绪: {db_path}")
    else:
        print(f"  数据库将自动创建")

    print(f"\n启动服务器: http://localhost:8000")
    print(f"API 文档: http://localhost:8000/docs")
    print(f"前端界面: http://localhost:8000/static/index.html")
    print("=" * 60)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
