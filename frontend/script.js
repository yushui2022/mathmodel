// DOM元素
const problemInput = document.getElementById('problem-input');
const submitBtn = document.getElementById('submit-btn');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const resultsContainer = document.getElementById('results-container');
const chartsContainer = document.getElementById('charts-container');
const webDataContainer = document.getElementById('web-data-container');
const modeSingle = document.getElementById('mode-single');
const modeMulti = document.getElementById('mode-multi');
const conversationWrapper = document.getElementById('conversation-wrapper');
const conversationInput = document.getElementById('conversation-input');

// 标签切换功能
function setupTabs() {
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // 更新标签按钮状态
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 更新标签内容状态
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-content`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// 更新进度条
function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
    const percentNum = Math.max(0, Math.min(100, Math.round(percent)));
    const percentLabel = document.getElementById('progress-percent');
    if (percentLabel) {
        percentLabel.textContent = `${percentNum}%`;
    }
    
    // 添加进度日志
    addProgressLog(text);
}

// 更新步骤指示（简单分段）
function updateSteps(percent) {
    const steps = document.querySelectorAll('.step-dot');
    const p = Number(percent) || 0;
    steps.forEach((step, idx) => {
        const threshold = (idx + 1) * 25; // 4个节点
        if (p >= threshold) {
            step.classList.add('done');
        } else {
            step.classList.remove('done');
        }
    });
}

// 添加进度日志
function addProgressLog(message) {
    const progressLog = document.getElementById('progress-log');
    if (!progressLog) return;
    
    const logItem = document.createElement('div');
    logItem.className = 'progress-log-item';
    
    // 添加时间戳
    const timestamp = new Date().toLocaleTimeString();
    const timeSpan = document.createElement('span');
    timeSpan.className = 'progress-log-time';
    timeSpan.textContent = timestamp;
    
    // 添加消息内容
    const messageSpan = document.createElement('span');
    messageSpan.className = 'progress-log-message';
    messageSpan.textContent = message;
    
    logItem.appendChild(timeSpan);
    logItem.appendChild(messageSpan);
    
    // 添加到日志容器
    progressLog.appendChild(logItem);
    
    // 自动滚动到底部
    progressLog.scrollTop = progressLog.scrollHeight;
}

// 显示生成结果
function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    if (!results || Object.keys(results).length === 0) {
        resultsContainer.innerHTML = '<p>暂无生成结果</p>';
        return;
    }
    
    // 论文章节顺序
    const sections = [
        { key: 'abstract', title: '摘要' },
        { key: 'problem_restatement', title: '问题重述' },
        { key: 'problem_analysis', title: '问题分析' },
        { key: 'model_assumption', title: '模型假设' },
        { key: 'symbol_explanation', title: '符号说明' },
        { key: 'model_building', title: '模型建立' },
        { key: 'model_solving', title: '模型求解' },
        { key: 'results_analysis', title: '结果分析' }
    ];
    
    sections.forEach(section => {
        if (results[section.key]) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'result-section';
            sectionDiv.innerHTML = `
                <h4>${section.title}</h4>
                <p>${results[section.key]}</p>
            `;
            resultsContainer.appendChild(sectionDiv);
        }
    });
}

// 显示图表
function displayCharts(charts) {
    chartsContainer.innerHTML = '';
    
    if (!charts || charts.length === 0) {
        chartsContainer.innerHTML = '<p>暂无生成图表</p>';
        return;
    }
    
    const chartGrid = document.createElement('div');
    chartGrid.className = 'chart-grid';
    
    charts.forEach(chart => {
        const chartItem = document.createElement('div');
        chartItem.className = 'chart-item';
        chartItem.innerHTML = `
            <img src="../charts/${chart.file}" alt="${chart.name}">
            <p>${chart.name}</p>
        `;
        chartGrid.appendChild(chartItem);
    });
    
    chartsContainer.appendChild(chartGrid);
}

// 显示网络数据
function displayWebData(webData) {
    webDataContainer.innerHTML = '';
    
    if (!webData || Object.keys(webData).length === 0) {
        webDataContainer.innerHTML = '<p>暂无网络数据</p>';
        return;
    }

    // Summary
    if (webData.summary) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'web-data-item';
        summaryDiv.innerHTML = `
            <h4>抓取摘要</h4>
            <p>${webData.summary}</p>
        `;
        webDataContainer.appendChild(summaryDiv);
    }

    // Sources (with links)
    const sources = webData.sources || [];
    if (sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'web-data-item';
        sourcesDiv.innerHTML = `<h4>权威来源链接</h4>`;

        const ul = document.createElement('ul');
        ul.className = 'web-sources-list';
        sources.forEach(s => {
            const li = document.createElement('li');
            const title = s.title || s.url || '来源';
            const url = s.url || '';
            li.innerHTML = url
                ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>`
                : `${title}`;
            ul.appendChild(li);
        });
        sourcesDiv.appendChild(ul);
        webDataContainer.appendChild(sourcesDiv);
    }

    // Tables
    const tables = webData.tables || [];
    if (tables.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'web-data-item';
        emptyDiv.innerHTML = '<h4>抓取表格</h4><p>未抓取到可解析表格（可能页面无表格或被反爬）。</p>';
        webDataContainer.appendChild(emptyDiv);
        return;
    }

    tables.forEach((t, idx) => {
        const item = document.createElement('div');
        item.className = 'web-data-item';

        const sourceUrl = t.source_url || '';
        const preview = t.preview || {};
        const columns = preview.columns || [];
        const rows = preview.rows || [];
        const shape = preview.shape || [];

        // csv 下载链接（后端静态挂载 /tavily_runs）
        let csvLinkHtml = '';
        if (webData.run_dir && t.saved_path) {
            const downloadUrl = `http://localhost:8000/tavily_runs/${t.saved_path}`;
            csvLinkHtml = `<p><strong>CSV下载：</strong><a href="${downloadUrl}" target="_blank" rel="noopener noreferrer">${downloadUrl}</a></p>`;
        }

        item.innerHTML = `
            <h4>表格 ${idx + 1}${shape.length ? `（原始维度：${shape[0]}×${shape[1]}）` : ''}</h4>
            ${sourceUrl ? `<p><strong>来源页：</strong><a href="${sourceUrl}" target="_blank" rel="noopener noreferrer">${sourceUrl}</a></p>` : ''}
            ${csvLinkHtml}
        `;

        const tableEl = document.createElement('table');
        tableEl.className = 'web-table';
        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        columns.forEach(c => {
            const th = document.createElement('th');
            th.textContent = c;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        tableEl.appendChild(thead);

        const tbody = document.createElement('tbody');
        rows.forEach(r => {
            const tr = document.createElement('tr');
            r.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        tableEl.appendChild(tbody);

        item.appendChild(tableEl);
        webDataContainer.appendChild(item);
    });

    // 原始/清洗后 CSV
    const rawCsvs = webData.raw_csvs || [];
    const cleanedCsvs = webData.cleaned_csvs || [];
    const runDir = webData.run_dir || '';

    const csvSection = document.createElement('div');
    csvSection.className = 'web-data-item';
    csvSection.innerHTML = `<h4>外部表格文件</h4>`;

    const csvList = document.createElement('div');
    csvList.className = 'csv-list';

    const buildLinks = (title, files) => {
        const block = document.createElement('div');
        block.className = 'csv-block';
        const h = document.createElement('h5');
        h.textContent = title;
        block.appendChild(h);
        if (!files || files.length === 0) {
            const p = document.createElement('p');
            p.textContent = '暂无';
            block.appendChild(p);
        } else {
            const ul = document.createElement('ul');
            files.forEach(f => {
                const li = document.createElement('li');
                const url = runDir ? `http://localhost:8000/tavily_runs/${runDir}/${f}` : f;
                li.innerHTML = `<a href="${url}" target="_blank" rel="noopener noreferrer">${f}</a>`;
                ul.appendChild(li);
            });
            block.appendChild(ul);
        }
        return block;
    };

    csvList.appendChild(buildLinks('爬取原始 CSV', rawCsvs));
    csvList.appendChild(buildLinks('清洗后 CSV', cleanedCsvs));

    csvSection.appendChild(csvList);
    webDataContainer.appendChild(csvSection);
}

// 生成过程 - 连接后端API
async function generatePaper(problem) {
    submitBtn.disabled = true;
    
    try {
        // 显示初始状态
        updateProgress(0, '开始生成...');
        
        // 启动进度轮询
        // 启动进度轮询（更轻量：进度条+步骤状态）
        const progressInterval = setInterval(async () => {
            try {
                const statusResponse = await fetch('http://localhost:8000/status');
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    updateProgress(statusData.progress, statusData.message);
                    updateSteps(statusData.progress);
                }
            } catch (error) {
                console.error('获取进度失败:', error);
            }
        }, 800); // 加快一点感知
        
        // 根据界面选择是否启用多轮对话
        const useMulti = modeMulti.checked;
        const conversationContext = useMulti ? (conversationInput.value || '') : '';

        // 发送生成请求
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                problem: problem,
                conversation_context: conversationContext
            })
        });
        
        // 清除进度轮询
        clearInterval(progressInterval);
        
        if (!response.ok) {
            throw new Error(`生成失败：${await response.text()}`);
        }
        
        const result = await response.json();
        
        // 显示完成状态
        updateProgress(100, '生成完成！');
        
        // 显示结果
        displayResults(result.results);
        displayCharts(result.charts);
        displayWebData(result.web_data);
        
    } catch (error) {
        updateProgress(0, `生成失败：${error.message}`);
        console.error('生成失败:', error);
    } finally {
        submitBtn.disabled = false;
    }
}

// 初始化
function init() {
    setupTabs();
    
    // 多轮对话切换显示补充输入框
    modeSingle.addEventListener('change', () => {
        if (modeSingle.checked) conversationWrapper.classList.add('hidden');
    });
    modeMulti.addEventListener('change', () => {
        if (modeMulti.checked) conversationWrapper.classList.remove('hidden');
    });
    // 默认单轮 -> 隐藏补充输入
    conversationWrapper.classList.add('hidden');
    
    // 提交按钮点击事件
    submitBtn.addEventListener('click', async () => {
        const problem = problemInput.value.trim();
        
        if (!problem) {
            alert('请输入数学建模题目！');
            return;
        }
        
        // 开始生成过程
        await generatePaper(problem);
    });

    // 初始步骤状态
    updateSteps(0);
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', init);
