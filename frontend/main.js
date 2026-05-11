/**
 * 蛋白质判别系统 - 前端交互逻辑
 * 处理文件上传、API 调用、结果展示、搜索等功能
 */

// ============================================================
// 全局状态
// ============================================================

const API_BASE = '';  // API 基础路径，生产环境可配置
let currentTab = 'predict';
let selectedFile = null;
let browsePage = 1;
const PAGE_SIZE = 20;

// 3D 查看器状态
let viewerInstance = null;
let viewerCurrentPdb = null;

// ============================================================
// DOM 元素引用缓存
// ============================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================================
// 初始化
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initSearch();
    initBrowse();
    initViewer();
    checkHealth();
});

// ============================================================
// 导航切换
// ============================================================

function initNavigation() {
    const links = $$('.nav-link');
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tab) {
    currentTab = tab;

    // 更新导航
    $$('.nav-link').forEach(l => {
        l.classList.toggle('active', l.dataset.tab === tab);
    });

    // 更新面板
    $$('.panel').forEach(p => p.classList.remove('active'));
    const panel = $(`#${tab}-panel`);
    if (panel) panel.classList.add('active');

    // 加载对应数据
    if (tab === 'browse') loadBrowseData();
    if (tab === 'search') $('#searchInput')?.focus();
}

// ============================================================
// 健康检查
// ============================================================

async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        console.log('服务状态:', data);
    } catch (err) {
        console.warn('后端服务未连接:', err.message);
    }
}

// ============================================================
// 3D 蛋白质结构查看器
// ============================================================

function initViewer() {
    $('#btnView3D').addEventListener('click', openViewer);
    $('#viewerCloseBtn').addEventListener('click', closeViewer);

    $('#viewerOverlay').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeViewer();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && viewerInstance) closeViewer();
    });

    $$('.viewer-style-btn[data-style]').forEach(b => {
        b.addEventListener('click', () => switchViewerStyle(b.dataset.style));
    });

    $('#viewerColorSelect').addEventListener('change', (e) => {
        switchViewerColor(e.target.value);
    });

    $('#viewerResetBtn').addEventListener('click', resetViewerCamera);
    $('#viewerRetryBtn').addEventListener('click', openViewer);
}

// ============================================================
// 文件上传处理
// ============================================================

function initUpload() {
    const uploadArea = $('#uploadArea');
    const fileInput = $('#fileInput');
    const selectFileBtn = $('#selectFileBtn');
    const clearFileBtn = $('#clearFileBtn');
    const predictBtn = $('#predictBtn');

    // 点击选择文件
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // 清除文件
    clearFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // 预测
    predictBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (selectedFile) runPrediction();
    });
}

function handleFileSelect(file) {
    const validExtensions = ['.pdb', '.pdb.gz', '.ent', '.cif'];
    const filename = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => filename.endsWith(ext));

    if (!isValid) {
        showError('不支持的文件格式。请上传 .pdb, .pdb.gz, .ent, 或 .cif 文件。');
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        showError('文件大小超过 100MB 限制。');
        return;
    }

    selectedFile = file;
    hideError();

    // 更新 UI
    $('.upload-content').style.display = 'none';
    $('#filePreview').style.display = 'flex';
    $('#fileName').textContent = file.name;
    $('#fileSize').textContent = formatFileSize(file.size);
    $('#resultCard').style.display = 'none';
}

function clearFile() {
    selectedFile = null;
    $('#fileInput').value = '';
    $('.upload-content').style.display = '';
    $('#filePreview').style.display = 'none';
    $('#resultCard').style.display = 'none';
    $('#errorCard').style.display = 'none';
}

// ============================================================
// 预测逻辑
// ============================================================

async function runPrediction() {
    if (!selectedFile) return;

    // 显示加载状态
    $('#resultCard').style.display = 'none';
    $('#errorCard').style.display = 'none';
    $('#loadingContainer').style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const resp = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData,
        });

        $('#loadingContainer').style.display = 'none';

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '预测请求失败');
        }

        const result = await resp.json();
        displayResult(result);
    } catch (err) {
        $('#loadingContainer').style.display = 'none';
        showError(err.message || '预测过程中发生未知错误');
    }
}

function displayResult(result) {
    const card = $('#resultCard');
    if (!card) {
        console.error('displayResult: #resultCard not found in DOM');
        return;
    }
    card.style.display = 'block';

    // 基本信息
    $('#resultName').textContent = result.name || '-';
    $('#resultNameCN').textContent = translateProteinName(result.name || '');
    $('#resultCategory').textContent = result.category || '-';
    $('#resultCategoryCN').textContent = result.category_cn || result.category_cn_display || '-';
    $('#resultPdbId').textContent = result.pdb_id || '-';
    $('#resultSeqLen').textContent = result.sequence_length || '-';

    // 置信度
    const confidence = result.confidence;
    const badge = $('#confidenceBadge');
    if (badge) {
        if (confidence != null) {
            badge.textContent = `置信度 ${(confidence * 100).toFixed(1)}%`;
            badge.className = 'confidence-badge';
            if (confidence >= 0.8) badge.classList.add('confidence-high');
            else if (confidence >= 0.5) badge.classList.add('confidence-medium');
            else badge.classList.add('confidence-low');
        } else {
            badge.textContent = '已知分类';
            badge.className = 'confidence-badge known-label';
        }
    }

    // 描述
    const desc = result.additional_info || generateDescription(result);
    $('#resultDescription').textContent = desc;

    // 概率分布
    if (result.all_probabilities) {
        displayProbabilityBars(result.all_probabilities, result.category_cn);
    }

    // 更新 3D 查看器按钮关联的蛋白质信息
    const btn3D = $('#btnView3D');
    if (btn3D) {
        btn3D.dataset.proteinId = result.id || result.db_id || '';
        btn3D.dataset.pdbId = result.pdb_id || '';
        btn3D.dataset.proteinName = result.name || '';
    }

    // 滚动到结果
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayProbabilityBars(probs, predictedCategory) {
    const container = $('#probabilityBars');
    container.innerHTML = '';

    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([category, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-bar-item';

        const pct = (prob * 100).toFixed(1);
        const isHighlight = category === predictedCategory;

        item.innerHTML = `
            <span class="prob-bar-label">${category}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isHighlight ? 'highlight' : ''}"
                     style="width: ${Math.max(pct, 3)}%">
                    ${pct}%
                </div>
            </div>
        `;
        container.appendChild(item);
    });
}

function generateDescription(result) {
    const name = result.name || '未知蛋白质';
    const category = result.category_cn || '未知';
    const pdbId = result.pdb_id || '';
    return `${name} 被分类为「${category}」，PDB ID: ${pdbId}。`;
}

// ============================================================
// 搜索功能
// ============================================================

let searchDebounceTimer = null;

function initSearch() {
    const searchInput = $('#searchInput');
    const categoryFilter = $('#categoryFilter');

    // 输入时自动搜索 (300ms 防抖)
    searchInput.addEventListener('input', () => {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = setTimeout(performSearch, 300);
    });

    // 回车立即搜索
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(searchDebounceTimer);
            performSearch();
        }
    });

    categoryFilter.addEventListener('change', performSearch);
    $('#searchBtn').addEventListener('click', performSearch);
}

async function performSearch() {
    const query = $('#searchInput').value.trim();
    const category = $('#categoryFilter').value;
    const container = $('#searchResults');
    const emptyMsg = $('#searchEmpty');
    const statsBar = $('#searchStats');

    if (!query && !category) {
        container.innerHTML = '';
        if (statsBar) statsBar.style.display = 'none';
        emptyMsg.style.display = '';
        emptyMsg.querySelector('p').textContent = '输入关键词或选择分类开始搜索';
        return;
    }

    container.innerHTML = '<div class="loading-container"><div class="spinner"></div><p>搜索中...</p></div>';
    emptyMsg.style.display = 'none';
    if (statsBar) statsBar.style.display = 'none';

    const params = new URLSearchParams();
    if (query) params.set('q', query);
    if (category) params.set('category', category);
    params.set('limit', '100');

    try {
        const resp = await fetch(`${API_BASE}/search?${params}`);
        if (!resp.ok) throw new Error('搜索请求失败');
        const data = await resp.json();

        // 显示统计
        if (!statsBar) {
            const bar = document.createElement('div');
            bar.id = 'searchStats';
            bar.className = 'search-stats';
            container.parentNode.insertBefore(bar, container);
        }
        const sb = $('#searchStats');
        sb.style.display = 'flex';
        sb.innerHTML = `<span>找到 <strong>${data.count}</strong> 条结果</span>
                        ${query ? `<span>关键词: <em>${escapeHtml(query)}</em></span>` : ''}
                        ${category ? `<span>分类: <em>${escapeHtml(category)}</em></span>` : ''}`;

        if (data.results.length === 0) {
            container.innerHTML = '';
            emptyMsg.style.display = '';
            emptyMsg.querySelector('p').textContent = '未找到匹配的蛋白质记录';
            return;
        }

        container.innerHTML = data.results.map(protein => {
            const desc = protein.additional_info || '';
            const descPreview = desc.length > 120 ? desc.substring(0, 120) + '...' : desc;
            return `
            <div class="search-result-card" onclick="showProteinDetail(${protein.id})" title="点击查看详情">
                <div class="search-result-header">
                    <span class="search-result-name">${escapeHtml(protein.name)}</span>
                    <span class="search-result-category">${escapeHtml(protein.category_cn)}</span>
                </div>
                <div class="search-result-meta">
                    <span>PDB: ${escapeHtml(protein.pdb_id || '-')}</span>
                    <span>置信度: ${protein.confidence != null ? (protein.confidence * 100).toFixed(1) + '%' : '已知分类'}</span>
                    <span>${formatDate(protein.created_at)}</span>
                </div>
                ${descPreview ? `<div class="search-result-desc">${escapeHtml(descPreview)}</div>` : ''}
            </div>
        `}).join('');
    } catch (err) {
        container.innerHTML = '';
        showToast('搜索失败: ' + err.message, 'error');
    }
}

async function showProteinDetail(id) {
    try {
        const resp = await fetch(`${API_BASE}/protein/${id}`);
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `请求失败 (HTTP ${resp.status})`);
        }
        const protein = await resp.json();

        switchTab('predict');

        try {
            displayResult({
                id: protein.id,
                name: protein.name,
                category: protein.category,
                category_cn: protein.category_cn,
                pdb_id: protein.pdb_id,
                sequence_length: protein.sequence_length,
                confidence: protein.confidence,
                additional_info: protein.additional_info,
                pdb_file_path: protein.pdb_file_path,
            });
        } catch (displayErr) {
            console.error('displayResult error:', displayErr);
            // 结果卡片已显示, 仅部分字段可能未填充 — 不弹出错误
        }
    } catch (err) {
        console.error('showProteinDetail error:', err);
        showToast('获取蛋白质详情失败: ' + err.message, 'error');
    }
}

// ============================================================
// 浏览功能
// ============================================================

function initBrowse() {
    // 初始化时加载
}

async function loadBrowseData(page = 1) {
    browsePage = page;
    const offset = (page - 1) * PAGE_SIZE;

    try {
        const resp = await fetch(`${API_BASE}/proteins?limit=${PAGE_SIZE}&offset=${offset}`);
        if (!resp.ok) throw new Error('获取数据失败');
        const data = await resp.json();

        renderStats(data.total);
        renderBrowseTable(data.results);
        renderPagination(data.total, page);
    } catch (err) {
        console.error('加载浏览数据失败:', err);
    }
}

function renderStats(total) {
    const container = $('#statsCards');
    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-number">${total}</div>
            <div class="stat-label">总记录数</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">分类类别</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">GNN</div>
            <div class="stat-label">模型类型</div>
        </div>
    `;
}

function renderBrowseTable(proteins) {
    const tbody = $('#browseTableBody');
    if (proteins.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:32px;color:#9ca3af;">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = proteins.map(p => `
        <tr class="browse-row" onclick="showProteinDetail(${p.id})" title="点击查看详情">
            <td>${p.id}</td>
            <td><strong>${escapeHtml(p.name)}</strong></td>
            <td><span class="category-tag">${escapeHtml(p.category_cn)}</span></td>
            <td>${escapeHtml(p.category)}</td>
            <td>${escapeHtml(p.pdb_id || '-')}</td>
            <td class="confidence-cell">${p.confidence != null ? (p.confidence * 100).toFixed(1) + '%' : '<span class="known-label">已知分类</span>'}</td>
            <td>${formatDate(p.created_at)}</td>
        </tr>
    `).join('');
}

function renderPagination(total, currentPage) {
    const container = $('#pagination');
    const totalPages = Math.ceil(total / PAGE_SIZE);

    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    const WINDOW = 1; // 当前页前后各显示1页

    let html = '<span class="page-group">';

    // 上一页
    html += `<button class="page-nav" onclick="loadBrowseData(${currentPage - 1})"
              ${currentPage === 1 ? 'disabled' : ''} title="上一页">‹</button>`;

    // 第一页
    if (currentPage > WINDOW + 2) {
        html += `<button onclick="loadBrowseData(1)">1</button>`;
        html += `<span class="page-ellipsis">…</span>`;
    } else if (currentPage > WINDOW + 1) {
        html += `<button onclick="loadBrowseData(1)">1</button>`;
    }

    // 窗口内的页码
    const start = Math.max(2, currentPage - WINDOW);
    const end = Math.min(totalPages - 1, currentPage + WINDOW);
    for (let i = start; i <= end; i++) {
        const active = i === currentPage ? ' active' : '';
        html += `<button class="${active}" onclick="loadBrowseData(${i})">${i}</button>`;
    }

    // 最后一页
    if (currentPage < totalPages - WINDOW - 1) {
        html += `<span class="page-ellipsis">…</span>`;
        html += `<button onclick="loadBrowseData(${totalPages})">${totalPages}</button>`;
    } else if (currentPage < totalPages - WINDOW) {
        html += `<button onclick="loadBrowseData(${totalPages})">${totalPages}</button>`;
    }

    // 下一页
    html += `<button class="page-nav" onclick="loadBrowseData(${currentPage + 1})"
              ${currentPage === totalPages ? 'disabled' : ''} title="下一页">›</button>`;

    html += '</span>';

    // 页码跳转
    html += `<span class="page-jump">
        <input type="number" id="pageJumpInput" min="1" max="${totalPages}"
               placeholder="${currentPage}" title="输入页码回车跳转"
               onkeydown="if(event.key==='Enter')jumpToPage(${totalPages})">
    </span>`;

    container.innerHTML = html;
}

function jumpToPage(totalPages) {
    const input = $('#pageJumpInput');
    const page = parseInt(input.value);
    if (page >= 1 && page <= totalPages) {
        loadBrowseData(page);
    } else {
        showToast(`请输入 1-${totalPages} 之间的页码`, 'error');
    }
}

// ============================================================
// 工具函数
// ============================================================

function showError(message) {
    const card = $('#errorCard');
    card.style.display = 'flex';
    $('#errorMessage').textContent = message;
    $('#resultCard').style.display = 'none';
}

function hideError() {
    $('#errorCard').style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    try {
        const d = new Date(dateStr);
        return d.toLocaleDateString('zh-CN', {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit'
        });
    } catch {
        return dateStr;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function translateProteinName(name) {
    if (!name) return '';
    // 简单翻译规则
    const translations = {
        '结合蛋白': 'Binding Protein',
        '复合物蛋白': 'Complex Protein',
        '核酸结合蛋白': 'Nucleic Acid Binding Protein',
        '核酸': 'Nucleic Acid',
    };

    // 如果已是中文格式，直接返回
    for (const cn of Object.keys(translations)) {
        if (name.startsWith(cn)) {
            return name;
        }
    }

    // 如果是英文名，尝试翻译
    let translated = name;
    if (translated.includes('Binding Protein')) translated = translated.replace('Binding Protein', '结合蛋白');
    if (translated.includes('Complex Protein')) translated = translated.replace('Complex Protein', '复合物蛋白');
    if (translated.includes('Nucleic Acid Binding')) translated = translated.replace('Nucleic Acid Binding', '核酸结合');

    return translated;
}

function showToast(message, type = 'info') {
    // 简单的 toast 实现
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: ${type === 'error' ? 'var(--danger)' : 'var(--gray-800)'};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        z-index: 1000;
        animation: slideUp 0.3s ease;
        box-shadow: var(--shadow-lg);
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================
// 3D 蛋白质结构查看器 — 核心逻辑
// ============================================================

function ensure3DmolLoaded() {
    return new Promise((resolve, reject) => {
        if (typeof $3Dmol !== 'undefined') { resolve(); return; }

        const scriptEl = $('#script-3dmol');
        if (scriptEl.src) {
            const start = Date.now();
            const check = setInterval(() => {
                if (typeof $3Dmol !== 'undefined') { clearInterval(check); resolve(); }
                else if (Date.now() - start > 15000) { clearInterval(check); reject(new Error('3Dmol.js 加载超时')); }
            }, 100);
            return;
        }

        scriptEl.src = scriptEl.dataset.src;
        scriptEl.onload = () => resolve();
        scriptEl.onerror = () => reject(new Error('3Dmol.js 库加载失败，请检查网络连接。'));
    });
}

async function openViewer() {
    const btn = $('#btnView3D');
    const proteinId = btn.dataset.proteinId;
    const pdbId = btn.dataset.pdbId;
    const proteinName = btn.dataset.proteinName;

    if (!proteinId) {
        showToast('无法获取蛋白质信息', 'error');
        return;
    }

    btn.disabled = true;

    const overlay = $('#viewerOverlay');
    overlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    $('#viewerTitle').textContent = proteinName || '蛋白质结构';
    $('#viewerPdbBadge').textContent = pdbId || 'N/A';
    $('#viewerFileSize').textContent = '';
    $('#viewer3dContainer').innerHTML = '';
    $('#viewerLoading').style.display = 'flex';
    $('#viewerError').style.display = 'none';

    try {
        await ensure3DmolLoaded();
    } catch (err) {
        showViewerError(err.message);
        btn.disabled = false;
        return;
    }

    try {
        const resp = await fetch(`${API_BASE}/protein/${proteinId}/pdb`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '获取PDB数据失败');
        }
        const data = await resp.json();
        viewerCurrentPdb = data.pdb_content;
        $('#viewerFileSize').textContent = formatFileSize(data.file_size || 0);
    } catch (err) {
        showViewerError(err.message || '获取蛋白质结构数据失败');
        btn.disabled = false;
        return;
    }

    try {
        renderProtein(viewerCurrentPdb);
        $('#viewerLoading').style.display = 'none';
        btn.disabled = false;
    } catch (err) {
        showViewerError('蛋白质结构渲染失败: ' + (err.message || '未知错误'));
        btn.disabled = false;
    }
}

function renderProtein(pdbContent) {
    const container = $('#viewer3dContainer');
    container.innerHTML = '';

    const viewerDiv = document.createElement('div');
    viewerDiv.id = 'viewer3d';
    viewerDiv.style.width = '100%';
    viewerDiv.style.height = '100%';
    viewerDiv.style.position = 'relative';
    container.appendChild(viewerDiv);

    viewerInstance = $3Dmol.createViewer('viewer3d', {
        backgroundColor: '0x1a1a2e',
        antialias: true,
    });

    viewerInstance.addModel(pdbContent, 'pdb');
    viewerInstance.setStyle({}, { cartoon: { color: 'spectrum' } });
    viewerInstance.zoomTo();
    viewerInstance.render();
}

function switchViewerStyle(style) {
    if (!viewerInstance) return;

    $$('.viewer-style-btn[data-style]').forEach(b => {
        b.classList.toggle('active', b.dataset.style === style);
    });

    viewerInstance.removeAllSurfaces();

    const colorScheme = $('#viewerColorSelect').value;
    const styleSpec = {};
    const colorObj = {};

    if (colorScheme === 'spectrum') {
        Object.assign(colorObj, { color: 'spectrum' });
    } else if (colorScheme === 'chain') {
        Object.assign(colorObj, { colors: { chain: 'spectrum' } });
    } else if (colorScheme === 'ss') {
        Object.assign(colorObj, { colorscheme: 'ssJmol' });
    } else if (colorScheme === 'amino') {
        Object.assign(colorObj, { colorscheme: 'amino' });
    }

    if (style === 'cartoon') {
        Object.assign(styleSpec, { cartoon: colorObj });
    } else if (style === 'stick') {
        Object.assign(styleSpec, { stick: Object.assign({}, colorObj, { radius: 0.15 }) });
    } else if (style === 'surface') {
        Object.assign(styleSpec, { surface: Object.assign({}, colorObj, { opacity: 0.85 }) });
    } else if (style === 'line') {
        Object.assign(styleSpec, { line: colorObj });
    }

    viewerInstance.setStyle({}, styleSpec);
    viewerInstance.render();
}

function switchViewerColor(color) {
    if (!viewerInstance) return;
    const activeStyle = document.querySelector('.viewer-style-btn[data-style].active');
    const style = activeStyle ? activeStyle.dataset.style : 'cartoon';

    viewerInstance.removeAllSurfaces();

    const styleSpec = {};
    const colorObj = {};

    if (color === 'spectrum') {
        Object.assign(colorObj, { color: 'spectrum' });
    } else if (color === 'chain') {
        Object.assign(colorObj, { colors: { chain: 'spectrum' } });
    } else if (color === 'ss') {
        Object.assign(colorObj, { colorscheme: 'ssJmol' });
    } else if (color === 'amino') {
        Object.assign(colorObj, { colorscheme: 'amino' });
    }

    if (style === 'cartoon') {
        Object.assign(styleSpec, { cartoon: colorObj });
    } else if (style === 'stick') {
        Object.assign(styleSpec, { stick: Object.assign({}, colorObj, { radius: 0.15 }) });
    } else if (style === 'surface') {
        Object.assign(styleSpec, { surface: Object.assign({}, colorObj, { opacity: 0.85 }) });
    } else if (style === 'line') {
        Object.assign(styleSpec, { line: colorObj });
    }

    viewerInstance.setStyle({}, styleSpec);
    viewerInstance.render();
}

function resetViewerCamera() {
    if (!viewerInstance) return;
    viewerInstance.zoomTo();
    viewerInstance.render();
}

function showViewerError(message) {
    $('#viewerLoading').style.display = 'none';
    $('#viewer3dContainer').innerHTML = '';
    const errEl = $('#viewerError');
    errEl.style.display = 'flex';
    $('#viewerErrorMessage').textContent = message;
}

function closeViewer() {
    if (viewerInstance) {
        try {
            viewerInstance.removeAllSurfaces();
            viewerInstance.removeAllModels();
            viewerInstance.clear();
        } catch (e) { /* ignore teardown errors */ }
        viewerInstance = null;
    }

    viewerCurrentPdb = null;
    $('#viewer3dContainer').innerHTML = '';
    $('#viewerOverlay').style.display = 'none';
    $('#viewerLoading').style.display = 'flex';
    $('#viewerError').style.display = 'none';
    $('#btnView3D').disabled = false;
    document.body.style.overflow = '';
}
