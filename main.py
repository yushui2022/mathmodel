import os
import time
from operator import itemgetter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import docx
from docx.shared import Inches
import numpy as np
import pandas as pd

# 导入自定义模块
from data_analysis import DataGenerator, StatisticalAnalyzer, DataProcessor
from visualization import ChartGenerator
from math_utils import OptimizationSolver, StatisticalModel, NumericalMethods, ModelEvaluator

# ===================== 1. 初始化本地组件（基础配置） =====================
# 初始化Ollama嵌入模型（本地运行）
embeddings = OllamaEmbeddings(
    model="bge-m3:latest",  # 本地嵌入模型（需先ollama pull bge-m3:latest）
    base_url="http://localhost:11434"
)

# 初始化Ollama对话模型（本地运行，建议用llama3/mistral等大模型保证生成质量）
llm = ChatOllama(
    model="mistral:7b-instruct-v0.3-q4_K_M",  # 需先ollama pull llama3:8b-instruct-q4_K_M
    base_url="http://localhost:11434",
    temperature=0.1,  # 低随机性保证格式严格
    num_ctx=4096,  # 增大上下文窗口，容纳长提示词
    # 限制单次输出长度，避免某些情况下“越写越多”导致等待很久
    num_predict=1024,
)

# 加载本地FAISS向量库（数学建模知识库，需提前构建）
# 注意：替换为你自己的FAISS索引文件夹路径
vectorstore = FAISS.load_local(
    "math_modeling_faiss_index",  # 你的数学建模知识库向量库路径
    embeddings,
    allow_dangerous_deserialization=True  # 本地运行需开启
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 检索5条相关知识库内容

# ===================== 2. 嵌入你的所有提示词（核心部分） =====================
# 2.1 全局Agent前置提示（核心身份定义）
GLOBAL_AGENT_PROMPT = """你作为数学建模比赛专家，在接收用户提供的赛题后，需自主决策并生成论文模板，请紧密围绕知识库评估适配性最优的模型，全力生成文档，过程中禁止反问，无需询问论文标题，直接自主拟定，标题要贴合所选模型！
严格遵循指定的字数、段落规范，严谨完成内容撰写，确保生成的内容结构清晰、可读性强，暂无法确定的内容留空并添加【需补充】标记提示读者。"""

# 2.2 摘要生成提示词（完整嵌入你的格式要求）
ABSTRACT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文摘要必须严格按照以下格式生成，字数总计800字左右：
### 摘要模板
**第一段100字：**
本文通过建立和分析数学模型，解决了[具体问题]，对[相关领域或实际应用]具有重要的理论和实践意义。

**第二段50字：**
在模型建立方面，针对问题一，我们构建了一个基于[模型或评价体系名称]的模型；针对问题二，我们发展了一个基于[模型或评价体系名称]的模型；对于问题三，我们提出了一个基于[模型或评价体系名称]的模型。

**第三段150字：**
针对问题一，我们建立了[具体模型名称]，运用[使用的软件名称]，采用了[使用的算法或方法]，计算了[需要计算的变量或参数]，进行了[预测/评价/分析等]，以[具体目的或效果]。

**第四段150字：**
针对问题二，我们构建了[具体模型名称]，利用[使用的软件名称]，实施了[使用的算法或方法]，分析了[需要分析的数据或现象]，完成了[预测/评价/分析等]，旨在[具体目的或效果]。

**第五段150字：**
针对问题三，我们发展了[具体模型名称]，通过[使用的软件名称]，应用了[使用的算法或方法]，评估了[需要评估的指标或影响]，实现了[预测/评价/分析等]，以期达到[具体目的或效果]。

**第六段100字：**
通过这些模型的建立和验证，我们得到了[具体结果或发现]。在分析模型的优缺点后，我们提出了[改进措施或建议]，以期提高模型的[准确性/实用性/效率]。
请严格按照上述分段和字数要求生成摘要，自主拟定论文标题，标题要贴合所选模型！"""),
    ("human", """请根据以下数学建模赛题相关信息生成摘要：
赛题相关信息：{user_prompt}
知识库参考内容：{context}
要求：摘要总字数800字左右，严格按指定分段格式生成，标题自主拟定且贴合模型！""")
])

# 2.3 问题重述生成提示词（嵌入你的格式要求）
PROBLEM_RESTATEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文问题重述必须严格按照以下格式生成：
**第一段300字: 1.1 问题背景:** 写出该问题出现的问题背景,可以从社会,时代,问题现实,问题原因等角度写,字数200-400字。
**第二段: 1.2 问题提出:** 重述赛题问题,可润色但严禁改变题意。
请基于已生成的摘要内容，严格按此格式生成问题重述！"""),
    ("human", """请根据以下摘要内容生成问题重述：
摘要内容：{abstract_content}
赛题相关信息：{user_prompt}
知识库参考内容：{context}
要求：严格按指定分段和字数要求生成！""")
])

# 2.4 问题分析生成提示词（嵌入你的格式要求）
PROBLEM_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文问题分析必须严格按照以下格式生成，要求：中肯、确切；术语专业、内行；原理依据正确、明确；表述简明，关键步骤列出；忌外行话、术语不明确、表述混乱冗长。
### 问题分析格式
**第一段(500字): 2.1 问题一分析:** 什么什么主要包括几个方面,根据什么什么小类的影响因素决定,建立怎样的模型或体系,提取了什么什么类的指标对数据进行特征训练(如果该问题运用到了学习算法请写出,如果没有使用,请不要写)。
**第二段(500字): 2.2 问题二分析:** 与第一段模板相似。
**第三段(500字): 2.3 问题三分析:** 与第一段模板相似。
请基于已生成的摘要和问题重述，严格按此格式生成问题分析！"""),
    ("human", """请根据以下内容生成问题分析：
摘要内容：{abstract_content}
问题重述内容：{problem_restatement_content}
赛题相关信息：{user_prompt}
知识库参考内容：{context}
要求：严格按指定分段和字数要求生成，术语专业、表述简明！""")
])

# 2.5 模型假设生成提示词（嵌入你的格式要求）
MODEL_ASSUMPTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文模型假设必须严格按照以下要求生成：
1. 共6条假设，每条不超过100字；
2. 假设需基于赛题条件和前文（摘要、问题分析）作出，如：假设XX与XX无关系、XX在XX情况下概率相同、XX仅受XX影响等；
3. 假设要贴合实际，符合数学建模逻辑。
请严格按此要求生成模型假设！"""),
    ("human", """请根据以下内容生成6条模型假设：
摘要内容：{abstract_content}
问题分析内容：{problem_analysis_content}
赛题相关信息：{user_prompt}
知识库参考内容：{context}
要求：每条假设≤100字，共6条，贴合赛题和前文内容！""")
])

# 2.6 符号说明生成提示词（嵌入你的格式要求）
SYMBOL_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文符号说明必须严格按照以下格式生成：
    格式要求："汉字"---"符号"，例如：收益---P；
    需包含赛题中的核心词语、建模中使用的矩阵/因素等符号定义；
    符号定义要专业、统一，符合数学建模规范。
    请基于已生成的摘要、问题分析，严格按此格式生成符号说明！"""),
    ("human", """请根据以下内容生成符号说明：
    摘要内容：{abstract_content}
    问题分析内容：{problem_analysis_content}
    赛题相关信息：{user_prompt}
    知识库参考内容：{context}
    要求：严格按"汉字---符号"格式生成，覆盖核心概念和建模变量！""")
])

# 2.7 模型建立生成提示词（新增）
MODEL_BUILDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文模型建立章节必须严格按照以下格式生成，字数总计2000字左右：
    ### 模型建立格式
    **第一段(600字): 3.1 问题一模型建立:** 
    详细描述针对问题一建立的数学模型，包括：
    1. 模型类型和理论基础（如线性回归、时间序列、优化模型等）；
    2. 模型数学表达式和参数说明；
    3. 模型建立的具体步骤和方法；
    4. 使用的算法和求解思路。
    
    **第二段(600字): 3.2 问题二模型建立:** 
    与第一段格式相似，详细描述问题二的模型建立过程。
    
    **第三段(600字): 3.3 问题三模型建立:** 
    与第一段格式相似，详细描述问题三的模型建立过程。
    
    **第四段(200字): 3.4 模型关联性分析:** 
    说明三个模型之间的关联关系，如何协同解决问题。
    
    要求：模型描述要专业、数学表达式要准确、逻辑清晰！"""),
    ("human", """请根据以下内容生成模型建立章节：
    摘要内容：{abstract_content}
    问题分析内容：{problem_analysis_content}
    模型假设内容：{model_assumption_content}
    符号说明内容：{symbol_explanation_content}
    赛题相关信息：{user_prompt}
    知识库参考内容：{context}
    数据分析结果：{data_analysis_results}
    要求：严格按指定分段和字数要求生成，包含数学表达式！""")
])

# 2.8 模型求解生成提示词（新增）
MODEL_SOLVING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文模型求解章节必须严格按照以下格式生成，字数总计1500字左右：
    ### 模型求解格式
    **第一段(500字): 4.1 问题一模型求解:** 
    详细描述问题一模型的求解过程，包括：
    1. 使用的求解算法（如梯度下降、线性规划、数值积分等）；
    2. 求解步骤和关键计算过程；
    3. 使用的软件工具（如Python、MATLAB、Excel等）；
    4. 求解结果和关键数值。
    
    **第二段(500字): 4.2 问题二模型求解:** 
    与第一段格式相似，描述问题二的求解过程。
    
    **第三段(500字): 4.3 问题三模型求解:** 
    与第一段格式相似，描述问题三的求解过程。
    
    要求：求解过程要详细、结果要准确、可重现！"""),
    ("human", """请根据以下内容生成模型求解章节：
    模型建立内容：{model_building_content}
    赛题相关信息：{user_prompt}
    知识库参考内容：{context}
    求解结果数据：{solving_results}
    要求：严格按指定分段和字数要求生成，包含具体求解步骤和结果！""")
])

# 2.9 结果分析生成提示词（新增）
RESULTS_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GLOBAL_AGENT_PROMPT),
    ("system", """数学建模论文结果分析章节必须严格按照以下格式生成，字数总计1500字左右：
    ### 结果分析格式
    **第一段(500字): 5.1 问题一结果分析:** 
    详细分析问题一的求解结果，包括：
    1. 结果的可视化展示（图表说明）；
    2. 结果的合理性和有效性分析；
    3. 关键发现和洞察；
    4. 结果的敏感性分析或误差分析。
    
    **第二段(500字): 5.2 问题二结果分析:** 
    与第一段格式相似，分析问题二的结果。
    
    **第三段(500字): 5.3 问题三结果分析:** 
    与第一段格式相似，分析问题三的结果。
    
    要求：分析要深入、图表要引用、结论要明确！"""),
    ("human", """请根据以下内容生成结果分析章节：
    模型求解内容：{model_solving_content}
    赛题相关信息：{user_prompt}
    知识库参考内容：{context}
    图表文件列表：{chart_files}
    统计分析结果：{statistical_results}
    要求：严格按指定分段和字数要求生成，引用生成的图表，进行深入分析！""")
])


# ===================== 3. 核心生成函数（LCEL链式调用） =====================
def format_docs(docs):
    """格式化FAISS检索的知识库内容"""
    # 控制检索上下文长度，避免 prompt 过长导致 Ollama 推理变慢
    max_doc_chars = 800
    max_total_chars = 3200
    parts = []
    total = 0
    for i, doc in enumerate(docs):
        content = (doc.page_content or "").strip().replace("\u0000", "")
        if len(content) > max_doc_chars:
            content = content[:max_doc_chars] + "…【已截断】"
        piece = f"参考资料{i + 1}：{content}"
        if total + len(piece) > max_total_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def _timed(label: str):
    """简单计时器：打印每一步耗时，方便定位卡顿点。"""
    start = time.time()

    def done(extra: str = ""):
        cost = time.time() - start
        tail = f" {extra}" if extra else ""
        print(f"    - {label} 耗时：{cost:.1f}s{tail}")

    return done


def generate_section(chain, input_data):
    """通用生成函数：调用LCEL链生成指定章节内容"""
    return chain.invoke(input_data).strip()


# 3.1 生成摘要
def generate_abstract(user_prompt):
    # 检索知识库相关内容
    t = _timed("摘要-检索")
    docs = retriever.invoke(user_prompt)
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    # 构建LCEL链（检索上下文 + 用户提示词 → 摘要生成）
    abstract_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"]
            )
            | ABSTRACT_PROMPT_TEMPLATE
            | llm
            | StrOutputParser()
    )
    t2 = _timed("摘要-生成")
    out = generate_section(abstract_chain, {"user_prompt": user_prompt})
    t2(f"(输出{len(out)}字)")
    return out


# 3.2 生成问题重述
def generate_problem_restatement(user_prompt, abstract_content):
    t = _timed("问题重述-检索")
    docs = retriever.invoke(user_prompt + "问题重述")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    restatement_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                abstract_content=lambda x: x["abstract_content"]
            )
            | PROBLEM_RESTATEMENT_PROMPT
            | llm
            | StrOutputParser()
    )
    t2 = _timed("问题重述-生成")
    out = generate_section(restatement_chain, {
        "user_prompt": user_prompt,
        "abstract_content": abstract_content
    })
    t2(f"(输出{len(out)}字)")
    return out


# 3.3 生成问题分析
def generate_problem_analysis(user_prompt, abstract_content, problem_restatement_content):
    t = _timed("问题分析-检索")
    docs = retriever.invoke(user_prompt + "问题分析")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    analysis_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                abstract_content=lambda x: x["abstract_content"],
                problem_restatement_content=lambda x: x["problem_restatement_content"]
            )
            | PROBLEM_ANALYSIS_PROMPT
            | llm
            | StrOutputParser()
    )
    t2 = _timed("问题分析-生成")
    out = generate_section(analysis_chain, {
        "user_prompt": user_prompt,
        "abstract_content": abstract_content,
        "problem_restatement_content": problem_restatement_content
    })
    t2(f"(输出{len(out)}字)")
    return out


# 3.4 生成模型假设
def generate_model_assumption(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("模型假设-检索")
    docs = retriever.invoke(user_prompt + "模型假设")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    assumption_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                abstract_content=lambda x: x["abstract_content"],
                problem_analysis_content=lambda x: x["problem_analysis_content"]
            )
            | MODEL_ASSUMPTION_PROMPT
            | llm
            | StrOutputParser()
    )
    t2 = _timed("模型假设-生成")
    out = generate_section(assumption_chain, {
        "user_prompt": user_prompt,
        "abstract_content": abstract_content,
        "problem_analysis_content": problem_analysis_content
    })
    t2(f"(输出{len(out)}字)")
    return out


# 3.5 生成符号说明
def generate_symbol_explanation(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("符号说明-检索")
    docs = retriever.invoke(user_prompt + "符号说明")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    symbol_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                abstract_content=lambda x: x["abstract_content"],
                problem_analysis_content=lambda x: x["problem_analysis_content"]
            )
            | SYMBOL_EXPLANATION_PROMPT
            | llm
            | StrOutputParser()
    )
    t2 = _timed("符号说明-生成")
    out = generate_section(symbol_chain, {
        "user_prompt": user_prompt,
        "abstract_content": abstract_content,
        "problem_analysis_content": problem_analysis_content
    })
    t2(f"(输出{len(out)}字)")
    return out


# 3.6 生成模型建立（新增）
def generate_model_building(user_prompt, abstract_content, problem_analysis_content,
                           model_assumption_content, symbol_explanation_content, data_analysis_results):
    docs = retriever.invoke(user_prompt + "模型建立")
    context = format_docs(docs)
    building_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                abstract_content=lambda x: x["abstract_content"],
                problem_analysis_content=lambda x: x["problem_analysis_content"],
                model_assumption_content=lambda x: x["model_assumption_content"],
                symbol_explanation_content=lambda x: x["symbol_explanation_content"],
                data_analysis_results=lambda x: x["data_analysis_results"]
            )
            | MODEL_BUILDING_PROMPT
            | llm
            | StrOutputParser()
    )
    return generate_section(building_chain, {
        "user_prompt": user_prompt,
        "abstract_content": abstract_content,
        "problem_analysis_content": problem_analysis_content,
        "model_assumption_content": model_assumption_content,
        "symbol_explanation_content": symbol_explanation_content,
        "data_analysis_results": data_analysis_results
    })


# 3.7 生成模型求解（新增）
def generate_model_solving(user_prompt, model_building_content, solving_results):
    docs = retriever.invoke(user_prompt + "模型求解")
    context = format_docs(docs)
    solving_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                model_building_content=lambda x: x["model_building_content"],
                solving_results=lambda x: x["solving_results"]
            )
            | MODEL_SOLVING_PROMPT
            | llm
            | StrOutputParser()
    )
    return generate_section(solving_chain, {
        "user_prompt": user_prompt,
        "model_building_content": model_building_content,
        "solving_results": solving_results
    })


# 3.8 生成结果分析（新增）
def generate_results_analysis(user_prompt, model_solving_content, chart_files, statistical_results):
    docs = retriever.invoke(user_prompt + "结果分析")
    context = format_docs(docs)
    analysis_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                user_prompt=lambda x: x["user_prompt"],
                model_solving_content=lambda x: x["model_solving_content"],
                chart_files=lambda x: x["chart_files"],
                statistical_results=lambda x: x["statistical_results"]
            )
            | RESULTS_ANALYSIS_PROMPT
            | llm
            | StrOutputParser()
    )
    return generate_section(analysis_chain, {
        "user_prompt": user_prompt,
        "model_solving_content": model_solving_content,
        "chart_files": chart_files,
        "statistical_results": statistical_results
    })


# 3.9 数据分析和可视化生成函数（新增）
def generate_data_and_charts(user_prompt, problem_type="general"):
    """
    根据问题类型生成示例数据和图表
    :param user_prompt: 用户提示词
    :param problem_type: 问题类型 ('time_series', 'optimization', 'regression', 'general')
    :return: (数据字典, 图表文件列表, 统计分析结果)
    """
    print("  生成示例数据和图表...")
    chart_generator = ChartGenerator()
    data_generator = DataGenerator()
    analyzer = StatisticalAnalyzer()
    
    generated_data = {}
    chart_files = []
    statistical_results = {}
    
    # 根据问题类型生成不同类型的数据和图表
    if "时间" in user_prompt or "预测" in user_prompt or "序列" in user_prompt:
        # 时间序列数据
        time_data = data_generator.generate_time_series_data(n_points=100, trend="linear", seasonality=True)
        generated_data['time_series'] = time_data
        
        # 生成时间序列图
        chart_file = chart_generator.plot_time_series(time_data, title="时间序列数据可视化")
        chart_files.append(("图1：时间序列数据可视化", chart_file))
        
        # 统计分析
        stats_result = analyzer.time_series_analysis(time_data)
        statistical_results['time_series'] = stats_result
    
    if "优化" in user_prompt or "分配" in user_prompt or "规划" in user_prompt:
        # 优化问题数据
        opt_data = data_generator.generate_optimization_data(n_items=20)
        generated_data['optimization'] = opt_data
        
        # 生成柱状图
        chart_file = chart_generator.plot_bar_chart(
            opt_data.head(10), '物品编号', '价值', title="物品价值分布"
        )
        chart_files.append(("图2：物品价值分布", chart_file))
    
    if "回归" in user_prompt or "关系" in user_prompt or "影响" in user_prompt:
        # 多变量数据
        multi_data = data_generator.generate_multivariate_data(n_samples=200, n_features=5)
        generated_data['multivariate'] = multi_data
        
        # 生成散点图
        chart_file = chart_generator.plot_scatter(
            multi_data, '特征1', '目标变量', title="特征与目标变量关系"
        )
        chart_files.append(("图3：特征与目标变量关系", chart_file))
        
        # 生成相关性热力图
        chart_file = chart_generator.plot_correlation_heatmap(multi_data, title="变量相关性热力图")
        chart_files.append(("图4：变量相关性热力图", chart_file))
        
        # 回归分析
        reg_result = analyzer.regression_analysis(multi_data, '特征1', '目标变量')
        statistical_results['regression'] = reg_result
    
    # 通用统计图表
    if len(generated_data) > 0:
        # 选择第一个数据集生成直方图
        first_data = list(generated_data.values())[0]
        numeric_cols = first_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            chart_file = chart_generator.plot_histogram(
                first_data, numeric_cols[0], title="数据分布直方图"
            )
            chart_files.append(("图5：数据分布直方图", chart_file))
            
            # 基本统计量
            stats_result = analyzer.basic_statistics(first_data, numeric_cols[0])
            statistical_results['basic_stats'] = stats_result
    
    # 格式化结果用于提示词
    chart_files_str = "\n".join([f"{name}: {file}" for name, file in chart_files])
    statistical_results_str = "\n".join([
        f"{key}: {value}" for key, value in statistical_results.items()
    ])
    
    return generated_data, chart_files, statistical_results_str


# ===================== 4. 整合生成Word文档 =====================
def generate_math_modeling_paper(user_prompt, save_path="数学建模论文.docx", 
                                generate_charts=True, generate_data=True):
    """
    生成完整的数学建模论文Word文档（增强版，包含数据分析和图表）
    :param user_prompt: 用户提供的赛题描述（几百字的长提示词）
    :param save_path: 论文保存路径
    :param generate_charts: 是否生成图表
    :param generate_data: 是否生成示例数据
    """
    print("=== 开始生成数学建模论文（增强版） ===")
    
    # 步骤0：生成示例数据和图表（如果需要）
    generated_data = {}
    chart_files = []
    statistical_results = ""
    data_analysis_results = ""
    
    if generate_data or generate_charts:
        generated_data, chart_files, statistical_results = generate_data_and_charts(user_prompt)
        data_analysis_results = f"已生成{len(generated_data)}组示例数据，{len(chart_files)}个图表"
    
    # 步骤1：生成摘要
    print("1/8 生成摘要...")
    abstract = generate_abstract(user_prompt)

    # 步骤2：生成问题重述
    print("2/8 生成问题重述...")
    problem_restatement = generate_problem_restatement(user_prompt, abstract)

    # 步骤3：生成问题分析
    print("3/8 生成问题分析...")
    problem_analysis = generate_problem_analysis(user_prompt, abstract, problem_restatement)

    # 步骤4：生成模型假设
    print("4/8 生成模型假设...")
    model_assumption = generate_model_assumption(user_prompt, abstract, problem_analysis)

    # 步骤5：生成符号说明
    print("5/8 生成符号说明...")
    symbol_explanation = generate_symbol_explanation(user_prompt, abstract, problem_analysis)

    # 步骤6：生成模型建立（新增）
    print("6/8 生成模型建立...")
    model_building = generate_model_building(
        user_prompt, abstract, problem_analysis, 
        model_assumption, symbol_explanation, data_analysis_results
    )
    
    # 步骤7：生成模型求解（新增）
    print("7/8 生成模型求解...")
    solving_results = f"使用Python和NumPy/SciPy进行数值计算，已生成{len(chart_files)}个可视化结果"
    model_solving = generate_model_solving(user_prompt, model_building, solving_results)
    
    # 步骤8：生成结果分析（新增）
    print("8/8 生成结果分析...")
    chart_files_str = "\n".join([f"{name}" for name, _ in chart_files])
    results_analysis = generate_results_analysis(
        user_prompt, model_solving, chart_files_str, statistical_results
    )

    # 步骤9：整合到Word文档
    print("整合内容到Word文档...")
    doc = docx.Document()

    # 添加标题（从摘要中提取，或自主生成）
    title = abstract.split("\n")[0].replace("**", "").strip() if "**" in abstract else "数学建模论文"
    doc.add_heading(title, level=0)

    # 添加摘要
    doc.add_heading("摘要", level=1)
    doc.add_paragraph(abstract)

    # 添加问题重述
    doc.add_heading("问题重述", level=1)
    doc.add_paragraph(problem_restatement)

    # 添加问题分析
    doc.add_heading("问题分析", level=1)
    doc.add_paragraph(problem_analysis)

    # 添加模型假设
    doc.add_heading("模型假设", level=1)
    doc.add_paragraph(model_assumption)

    # 添加符号说明
    doc.add_heading("符号说明", level=1)
    doc.add_paragraph(symbol_explanation)
    
    # 添加模型建立（新增）
    doc.add_heading("模型建立", level=1)
    doc.add_paragraph(model_building)
    
    # 添加模型求解（新增）
    doc.add_heading("模型求解", level=1)
    doc.add_paragraph(model_solving)
    
    # 添加结果分析（新增）
    doc.add_heading("结果分析", level=1)
    doc.add_paragraph(results_analysis)
    
    # 插入图表（新增）
    if generate_charts and len(chart_files) > 0:
        doc.add_heading("图表附录", level=1)
        for i, (chart_name, chart_file) in enumerate(chart_files, 1):
            if os.path.exists(chart_file):
                doc.add_paragraph(f"{chart_name}")
                doc.add_picture(chart_file, width=Inches(6))  # 设置图片宽度为6英寸
                doc.add_paragraph("")  # 添加空行

    # 保存文档
    doc.save(save_path)
    print(f"=== 论文生成完成！保存路径：{os.path.abspath(save_path)} ===")
    print(f"=== 共生成 {len(chart_files)} 个图表 ===")
    return save_path


# ===================== 5. 运行入口 =====================
if __name__ == "__main__":
    # 示例：用户输入的赛题长提示词（替换为你实际的赛题描述）
    USER_PROMPT = """
    赛题：电商平台销量预测与定价优化问题
    背景：某电商平台拥有近万种单品的销售数据，包含销量、价格、成本、品类等信息。平台希望通过数学建模解决以下问题：
    1. 分析销量的时间分布规律、品类间关联关系；
    2. 建立销量与定价的关系模型，制定定价与补货优化方案；
    3. 筛选核心单品，建立多目标优化模型最大化收益与市场需求。
    要求：基于真实数据特征，选择适配的数学模型，完成建模与求解。
    """

    # 生成论文
    generate_math_modeling_paper(USER_PROMPT)