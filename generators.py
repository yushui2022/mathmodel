import asyncio
import time
from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import llm
from prompts import (
    ABSTRACT_PROMPT_TEMPLATE,
    PROBLEM_RESTATEMENT_PROMPT,
    PROBLEM_ANALYSIS_PROMPT,
    MODEL_ASSUMPTION_PROMPT,
    SYMBOL_EXPLANATION_PROMPT,
    MODEL_BUILDING_PROMPT,
    MODEL_SOLVING_PROMPT,
    RESULTS_ANALYSIS_PROMPT,
)
from retrieval import get_cached_retrieval, get_cached_retrieval_async
from data_analysis import DataGenerator, StatisticalAnalyzer, DataProcessor
from visualization import ChartGenerator
import numpy as np
import pandas as pd
from tqdm import tqdm


def _timed(label: str):
    start = time.time()

    def done(extra: str = ""):
        cost = time.time() - start
        tail = f" {extra}" if extra else ""
        tqdm.write(f"    - {label} 耗时：{cost:.1f}s{tail}")

    return done


def format_docs(docs):
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


def generate_section(chain, input_data):
    return chain.invoke(input_data).strip()


def generate_abstract(user_prompt):
    t = _timed("摘要-检索")
    docs = get_cached_retrieval(user_prompt)
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    abstract_chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
        )
        | ABSTRACT_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    t2 = _timed("摘要-生成")
    out = generate_section(abstract_chain, {"user_prompt": user_prompt})
    t2(f"(输出{len(out)}字)")
    return out


def generate_problem_restatement(user_prompt, abstract_content):
    t = _timed("问题重述-检索")
    docs = get_cached_retrieval(user_prompt + "问题重述")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
        )
        | PROBLEM_RESTATEMENT_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("问题重述-生成")
    out = generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


def generate_problem_analysis(user_prompt, abstract_content, problem_restatement_content):
    t = _timed("问题分析-检索")
    docs = get_cached_retrieval(user_prompt + "问题分析")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_restatement_content=lambda x: x["problem_restatement_content"],
        )
        | PROBLEM_ANALYSIS_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("问题分析-生成")
    out = generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_restatement_content": problem_restatement_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


def generate_model_assumption(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("模型假设-检索")
    docs = get_cached_retrieval(user_prompt + "模型假设")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_analysis_content=lambda x: x["problem_analysis_content"],
        )
        | MODEL_ASSUMPTION_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("模型假设-生成")
    out = generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_analysis_content": problem_analysis_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


def generate_symbol_explanation(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("符号说明-检索")
    docs = get_cached_retrieval(user_prompt + "符号说明")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_analysis_content=lambda x: x["problem_analysis_content"],
        )
        | SYMBOL_EXPLANATION_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("符号说明-生成")
    out = generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_analysis_content": problem_analysis_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


async def generate_model_assumption_async(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("模型假设-检索(异步)")
    docs = await get_cached_retrieval_async(user_prompt + "模型假设")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_analysis_content=lambda x: x["problem_analysis_content"],
        )
        | MODEL_ASSUMPTION_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("模型假设-生成(异步)")
    out = await asyncio.to_thread(
        generate_section,
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_analysis_content": problem_analysis_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


async def generate_symbol_explanation_async(user_prompt, abstract_content, problem_analysis_content):
    t = _timed("符号说明-检索(异步)")
    docs = await get_cached_retrieval_async(user_prompt + "符号说明")
    context = format_docs(docs)
    t(f"(命中{len(docs)}条)")
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_analysis_content=lambda x: x["problem_analysis_content"],
        )
        | SYMBOL_EXPLANATION_PROMPT
        | llm
        | StrOutputParser()
    )
    t2 = _timed("符号说明-生成(异步)")
    out = await asyncio.to_thread(
        generate_section,
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_analysis_content": problem_analysis_content,
        },
    )
    t2(f"(输出{len(out)}字)")
    return out


def generate_model_building(
    user_prompt,
    abstract_content,
    problem_analysis_content,
    model_assumption_content,
    symbol_explanation_content,
    data_analysis_results,
):
    docs = get_cached_retrieval(user_prompt + "模型建立")
    context = format_docs(docs)
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            abstract_content=lambda x: x["abstract_content"],
            problem_analysis_content=lambda x: x["problem_analysis_content"],
            model_assumption_content=lambda x: x["model_assumption_content"],
            symbol_explanation_content=lambda x: x["symbol_explanation_content"],
            data_analysis_results=lambda x: x["data_analysis_results"],
        )
        | MODEL_BUILDING_PROMPT
        | llm
        | StrOutputParser()
    )
    return generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "abstract_content": abstract_content,
            "problem_analysis_content": problem_analysis_content,
            "model_assumption_content": model_assumption_content,
            "symbol_explanation_content": symbol_explanation_content,
            "data_analysis_results": data_analysis_results,
        },
    )


def generate_model_solving(user_prompt, model_building_content, solving_results):
    docs = get_cached_retrieval(user_prompt + "模型求解")
    context = format_docs(docs)
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            model_building_content=lambda x: x["model_building_content"],
            solving_results=lambda x: x["solving_results"],
        )
        | MODEL_SOLVING_PROMPT
        | llm
        | StrOutputParser()
    )
    return generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "model_building_content": model_building_content,
            "solving_results": solving_results,
        },
    )


def generate_results_analysis(user_prompt, model_solving_content, chart_files, statistical_results):
    docs = get_cached_retrieval(user_prompt + "结果分析")
    context = format_docs(docs)
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            user_prompt=lambda x: x["user_prompt"],
            model_solving_content=lambda x: x["model_solving_content"],
            chart_files=lambda x: x["chart_files"],
            statistical_results=lambda x: x["statistical_results"],
        )
        | RESULTS_ANALYSIS_PROMPT
        | llm
        | StrOutputParser()
    )
    return generate_section(
        chain,
        {
            "user_prompt": user_prompt,
            "model_solving_content": model_solving_content,
            "chart_files": chart_files,
            "statistical_results": statistical_results,
        },
    )


def generate_data_and_charts(user_prompt, problem_type="general"):
    tqdm.write("  生成示例数据和图表...")
    chart_generator = ChartGenerator()
    data_generator = DataGenerator()
    analyzer = StatisticalAnalyzer()

    generated_data = {}
    chart_files = []
    statistical_results = {}

    if "时间" in user_prompt or "预测" in user_prompt or "序列" in user_prompt:
        time_data = data_generator.generate_time_series_data(n_points=100, trend="linear", seasonality=True)
        generated_data["time_series"] = time_data
        chart_file = chart_generator.plot_time_series(time_data, title="时间序列数据可视化")
        chart_files.append(("图1：时间序列数据可视化", chart_file))
        stats_result = analyzer.time_series_analysis(time_data)
        statistical_results["time_series"] = stats_result

    if "优化" in user_prompt or "分配" in user_prompt or "规划" in user_prompt:
        opt_data = data_generator.generate_optimization_data(n_items=20)
        generated_data["optimization"] = opt_data
        chart_file = chart_generator.plot_bar_chart(opt_data.head(10), "物品编号", "价值", title="物品价值分布")
        chart_files.append(("图2：物品价值分布", chart_file))

    if "回归" in user_prompt or "关系" in user_prompt or "影响" in user_prompt:
        multi_data = data_generator.generate_multivariate_data(n_samples=200, n_features=5)
        generated_data["multivariate"] = multi_data
        chart_file = chart_generator.plot_scatter(multi_data, "特征1", "目标变量", title="特征与目标变量关系")
        chart_files.append(("图3：特征与目标变量关系", chart_file))
        chart_file = chart_generator.plot_correlation_heatmap(multi_data, title="变量相关性热力图")
        chart_files.append(("图4：变量相关性热力图", chart_file))
        reg_result = analyzer.regression_analysis(multi_data, "特征1", "目标变量")
        statistical_results["regression"] = reg_result

    if len(generated_data) > 0:
        first_data = list(generated_data.values())[0]
        numeric_cols = first_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            chart_file = chart_generator.plot_histogram(first_data, numeric_cols[0], title="数据分布直方图")
            chart_files.append(("图5：数据分布直方图", chart_file))
            stats_result = analyzer.basic_statistics(first_data, numeric_cols[0])
            statistical_results["basic_stats"] = stats_result

    chart_files_str = "\n".join([f"{name}: {file}" for name, file in chart_files])
    statistical_results_str = "\n".join([f"{key}: {value}" for key, value in statistical_results.items()])

    return generated_data, chart_files, statistical_results_str


__all__ = [
    "format_docs",
    "generate_abstract",
    "generate_problem_restatement",
    "generate_problem_analysis",
    "generate_model_assumption",
    "generate_symbol_explanation",
    "generate_model_assumption_async",
    "generate_symbol_explanation_async",
    "generate_model_building",
    "generate_model_solving",
    "generate_results_analysis",
    "generate_data_and_charts",
]

