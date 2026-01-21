# 数学建模论文生成系统（LangChain + Ollama + FAISS）

本项目用于：输入数学建模赛题描述，自动生成包含**摘要 / 问题重述 / 问题分析 / 模型假设 / 符号说明 / 模型建立 / 模型求解 / 结果分析**的 Word 论文，并可选生成图表、联网抓取“权威表格”摘要。

## 功能概览

- **RAG 检索增强**：FAISS 向量召回 + 关键词匹配 + rerank（二次重排序）
- **论文生成**：按 `prompts.py` 中的模板严格输出中文段落结构
- **图表与示例数据**：自动生成时间序列/回归/优化等示例数据及可视化图
- **联网权威表格（可选）**：Tavily 搜索 + `pandas.read_html` 抓取表格 + SQLite 缓存
- **API + 前端**：FastAPI 提供生成接口，`frontend/` 提供简单交互页面

## 环境要求

- Python 3.10+（建议 3.10/3.11）
- 已安装并运行 [Ollama](https://ollama.com/)

## 依赖安装

```bash
pip install -r requirements.txt
```

## 配置（重要）

复制环境变量模板：

```bash
copy env.example .env
```

需要关注的变量：

- **`OLLAMA_BASE_URL`**：Ollama 地址，默认 `http://localhost:11434`（见 `config.py`）
- **`TAVILY_API_KEY`**：启用联网抓取时需要；不填则自动跳过联网步骤（见 `web_data.py`）

## Ollama 模型准备

本项目默认使用（可在 `config.py` 中调整）：

- LLM：`llama3:latest`
- Embedding：`bge-m3:latest`
- Rerank embedding：`qllama/bge-reranker-v2-m3:latest`

拉取示例：

```bash
ollama pull llama3
ollama pull bge-m3
ollama pull qllama/bge-reranker-v2-m3
```

## 快速开始：直接生成 Word

运行入口在 `main.py`（命令行简单交互，默认把输出保存在当前目录 `数学建模论文.docx`）：

```bash
python main.py
```

流程：
- 粘贴/输入赛题描述；留空则使用内置示例赛题。
- 询问是否启用多轮对话补充信息（y/n）；选 y 可输入补充文本。
- 直接生成 Word，保存为 `数学建模论文.docx`；图表在 `charts/` 生成（若启用）。

## （可选）构建/更新知识库索引

默认检索索引由 `config.py` 加载：`math_modeling_faiss_index/`。

如果你想用自己的文本构建索引，可参考 `ingestion.py`（已改为读取项目内 `mediumblog1.txt`）：

```bash
python ingestion.py
```

会生成目录 `faiss_index_mediumblog/`（示例）。如需让主流程使用它，请在 `config.py` 中把 `FAISS.load_local(...)` 的目录改为新索引目录。

## 启动后端 API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

接口：

- `POST /generate`：提交题目，触发论文生成  
  请求体示例：
  ```json
  {
    "problem": "赛题长描述……",
    "conversation_context": "（可选）多轮对话补充信息，若提供会影响扩写与细节"
  }
  ```
- `GET /status`：轮询生成进度

## 启动前端（静态页面）

`frontend/` 是纯静态页面，你可以直接用浏览器打开 `frontend/index.html`。

如需本地静态服务器（可选）：

```bash
python -m http.server 5173 -d frontend
```

然后访问 `http://localhost:5173`。

### 前端多轮对话快捷提示
- 前端页面已有单/多轮切换：选择“多轮对话”时会展示补充信息文本框，填入内容后会自动作为 `conversation_context` 传给后端；选择“单轮对话”则不提交补充信息。
- “网络数据”板块现在会显示 Tavily 抓取的**原始 CSV**和**清洗后 CSV**下载链接（来自 `/tavily_runs/<时间戳>/processed/`），以及表格预览和权威来源链接。

## 目录结构（核心）

下面是“主流程”和“每个文件/目录职责”的详细说明，方便你把项目交给其他人也能快速上手。

## 整体流程（从输入到论文）

1. **输入赛题**
   - 前端：`frontend/index.html` 输入赛题文本，选择单轮/多轮，点“开始生成”
   - 后端：直接 `python main.py`，按提示粘贴赛题（可选多轮补充信息）

2. **（可选）联网抓取权威表格**
   - `web_data.py`：Tavily 搜索定位权威来源页面
   - `pandas.read_html`：抓网页表格并转 CSV
   - 落盘到 `tavily_runs/<时间戳>/tables/*.csv`，并写 `payload.json`

3. **通用表格处理（LLM 生成代码 → 子进程执行）**
   - `tavily_processing.py`：
     - 生成并执行“通用清洗代码”，得到 `cleaned_table_*.csv`
     - 生成并执行“通用可视化代码”，得到 PNG
     - 生成“结构化分析文字”，供论文引用
   - 处理结果落盘到：`tavily_runs/<时间戳>/processed/`

4. **生成论文（带迭代扩写到目标字数）**
   - `main.py` 生成各章节并写 Word：`数学建模论文.docx`
   - `iterative_generation.py` 对长章节（模型建立/求解/结果分析）做多轮扩写，确保长度达标

5. **前端展示**
   - 结果/图表/网络数据来源链接/表格预览
   - 网络数据附带“爬取前 CSV / 清洗后 CSV”下载链接（按时间戳分类）

## 文件与目录说明（逐个讲清楚）

### 入口与编排
- `main.py`
  - **作用**：核心编排入口（RAG 检索、Tavily 抓取、表格处理、各章节生成、Word 写入、进度回调）
  - **如何使用**：`python main.py`（会提示粘贴赛题，可选多轮补充信息）
  - **输出**：当前目录生成 `数学建模论文.docx`

- `pipeline.py`
  - **作用**：另一份论文生成编排（历史/备用实现）
  - **建议**：优先以 `main.py` 为主线维护

### 提示词与生成规范
- `prompts.py`
  - **作用**：论文各章节提示词模板（含“优化版”模型建立/求解/结果分析规则）

- `iterative_generation.py`
  - **作用**：通用“多轮扩写引擎”
  - **说明**：当长章节不达标时，会循环扩写 1~N 回合直到进入目标区间

### 模型/向量库配置
- `config.py`
  - **作用**：Ollama 地址、LLM、Embedding、Rerank embedding、FAISS 向量库加载
  - **注意**：默认加载 `math_modeling_faiss_index/`

### RAG 检索与知识库
- `ingestion.py`
  - **作用**：将文本切分后写入 FAISS（构建知识库索引）
  - **输入**：项目内 `mediumblog1.txt`（示例）

- `retrieval.py`
  - **作用**：混合检索与缓存（向量召回 + 关键词 + rerank）
  - **说明**：`main.py` 内也有一份混合检索实现；两者功能有重叠

- `math_modeling_faiss_index/`
  - **作用**：默认知识库索引（`index.faiss`/`index.pkl`）

### Tavily 联网抓取与落盘
- `web_data.py`
  - **作用**：Tavily 搜索 + 抓网页表格（`read_html`）+ SQLite 缓存 + 结果落盘
  - **落盘目录**：`tavily_runs/<时间戳>/`

- `web_cache.sqlite`
  - **作用**：联网缓存（避免重复搜索/重复抓取）

- `tavily_processing.py`
  - **作用**：对抓取表格做“通用清洗+分析+可视化”（LLM 生成代码并子进程执行）
  - **落盘目录**：`tavily_runs/<时间戳>/processed/`

- `tavily_runs/`
  - **作用**：每次抓取/处理的落盘根目录（按时间戳分类）

- `tavily_query_constraints.txt`
  - **作用**：Tavily 检索约束提示词（倾向权威/年鉴/官方数据表）

### 数据分析与可视化（示例数据）
- `data_analysis.py`：生成示例数据与统计分析（无真实数据时也能出图/出分析）
- `visualization.py`：绘图工具（折线/柱状/散点/热力/直方等）
- `charts/`：示例数据的图表输出目录（会插入 Word）

### 数学工具
- `math_utils.py`：优化/统计模型/数值方法/评估等工具集合

### API 与前端
- `api.py`
  - **作用**：FastAPI 服务（`/generate`、`/status`），并静态挂载 `/tavily_runs` 供 CSV 下载

- `frontend/`
  - `index.html`：输入区 + 单/多轮切换 + 结果/图表/网络数据展示
  - `script.js`：请求后端、轮询进度、渲染网页表格与 CSV 下载
  - `style.css`：样式

### 其它
- `mediumblog1.txt`：示例知识库文本
- `sample_prompt.txt`：示例赛题文本（便于演示/测试）

## 迭代扩写（长文控制）

- 模型建立 / 模型求解 / 结果分析 三个长章节，已接入“多轮扩写直到目标字数”：
  - 模型建立：目标约 2000 ±10% 字符
  - 模型求解：目标约 5000 ±10% 字符
  - 结果分析：目标约 4500 ±10% 字符
- 可选传入 `conversation_context`（通过 API 请求体）为扩写提供多轮对话上下文，模型会在扩写时引用这些补充信息。

## Roadmap / TODO

- ✅ 迭代扩写模块 + 接入三大长章节（长度自动达标）
- ✅ Tavily 抓取 → 清洗 → 分析 → 可视化 → Word/前端同步
- ✅ API 支持多轮对话上下文（`conversation_context`）
- ✅ 前端多轮对话 UI（单/多轮切换 + 补充信息框）
- ⏳ 扩写/修订专用提示词统一化（可进一步抽到 prompts.py）

## 常见问题

### 1) 运行时报 Ollama 连接失败

确认：

- Ollama 已启动
- `OLLAMA_BASE_URL` 正确（例如 `http://localhost:11434`）

### 2) 联网抓取没生效

只有设置了 `TAVILY_API_KEY` 才会启用联网抓取；否则会自动跳过并写入“未启用”提示。

## Roadmap / 进度

- ✅ 章节“迭代扩写”管线：模型建立/求解/结果分析按目标字数自动扩写到达标。
- ✅ Tavily 表格：自动清洗/分析/可视化，图表与生成代码一并落盘（`tavily_runs/latest/processed`）。
- ✅ API 支持可选的 `conversation_context`，可用于多轮补充信息再生成。
- ⏳ 前端多轮对话/历史上传：尚未接好，当前前端只支持单次输入。
- ⏳ 扩写专用提示词抽离到 `prompts.py`：当前在 `iterative_generation.py` 内部。


