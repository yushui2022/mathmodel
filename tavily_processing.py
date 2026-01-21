import os
import shutil
import subprocess
from typing import Dict, Any, List, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from config import llm


def _build_table_meta_from_csv_text(csv_text: str) -> str:
    """从 CSV 文本构造表格元信息，供 LLM 判读结构。"""
    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))
    desc = []
    desc.append(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    desc.append("列信息：")
    for col in df.columns:
        s = df[col]
        desc.append(
            f"- {col}: 非空 {s.notna().sum()} 条, 缺失 {s.isna().sum()} 条, "
            f"dtype={s.dtype}, 数值列? {pd.api.types.is_numeric_dtype(s)}"
        )
    return "\n".join(desc)


_CLEANING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是Python数据清洗专家，只输出可直接运行的Python代码，不要任何解释文字。"),
        (
            "human",
            """
已知有一份CSV数据，路径为：{input_path}
你需要：
1. 使用 pandas 读取；
2. 自动判断列类型，转换为合适的dtype（数值、类别、日期等）；
3. 处理缺失值：数值列用均值/中位数补全，类别列用众数或"未知"；
4. 删除完全重复行；
5. 对明显异常值（如3σ之外、或负值但按常识应为非负的量）进行简单处理（可截断到边界）；
6. 将清洗后的结果保存到：{output_path}，编码UTF-8，不要打印到屏幕。
约束：
- 只允许使用 pandas、numpy 和标准库；
- 不允许使用 os、subprocess、eval、exec 等危险函数；
- 程序入口写在 if __name__ == "__main__": 中，读写路径写死为上面给出的路径。

表结构元信息如下（仅供你判断类型和策略）：
{table_meta}

请只输出完整Python代码：
""",
        ),
    ]
)


def _gen_cleaning_code(table_meta: str, input_path: str, output_path: str) -> str:
    msgs = _CLEANING_PROMPT.format_messages(
        input_path=input_path.replace("\\", "\\\\"),
        output_path=output_path.replace("\\", "\\\\"),
        table_meta=table_meta,
    )
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))


_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是数学建模论文中的“数据分析”章节写作专家，输出中文分析报告，500-800字。",
        ),
        (
            "human",
            """
已完成对一张来自权威来源的表格的清洗，清洗后CSV路径为：{cleaned_path}。
请你假设已经读取了该表（不需要再写代码），根据下列元信息，写一段“可直接放进数学建模论文”的数据分析小节，包含：
1. 数据概况（样本量、指标数量、变量类型）；
2. 统计分析（均值、方差、分布形态、相关性等，可以定性+少量定量）；
3. 核心洞察（与建模目标相关的规律或发现）；
4. 建模建议（哪些变量可以作为自变量/因变量、可能适用的模型类型）。
要求：
- 500-800字，中文；
- 语气专业，适合作为论文中正式章节的一部分；
- 不要讨论具体代码或实现细节。

表格元信息如下：
{table_meta}
""",
        ),
    ]
)


def _gen_analysis_from_cleaned(cleaned_path: str, table_meta: str) -> str:
    msgs = _ANALYSIS_PROMPT.format_messages(
        cleaned_path=cleaned_path.replace("\\", "\\\\"),
        table_meta=table_meta,
    )
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))


_VIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是数据可视化专家，只输出Python代码，使用pandas和matplotlib保存PNG图。",
        ),
        (
            "human",
            """
有一份已经清洗好的CSV，路径：{cleaned_path}
请你生成Python代码，自动完成以下任务：
1. 使用 pandas 读取该CSV；
2. 自动检测数值列和类别列：
   - 对每个主要数值列画直方图（保存为 "hist_<列名>.png"）；
   - 对数值列之间画相关系数热力图（如果数值列>=3，保存为 "heatmap_corr.png"）；
   - 对类别列（取最多2列）画频数饼图或条形图（保存为 "cat_<列名>.png"）；
3. 所有图片保存到目录：{output_dir} 下；
4. 图片风格适合论文使用：字号适中，标题清晰，坐标轴标签完整。

约束：
- 只使用 pandas、numpy、matplotlib；
- 不使用 os、subprocess 等；
- 程序入口写在 if __name__ == "__main__": 中，路径写死为上述路径。

请只输出完整Python代码：
""",
        ),
    ]
)


def _gen_visualization_code(cleaned_path: str, output_dir: str) -> str:
    msgs = _VIZ_PROMPT.format_messages(
        cleaned_path=cleaned_path.replace("\\", "\\\\"),
        output_dir=output_dir.replace("\\", "\\\\"),
    )
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))


def _run_generated_code(code: str, code_path: str, timeout: int = 90) -> None:
    """将生成的代码写入文件，并以子进程方式运行。"""
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        subprocess.run(
            ["python", os.path.basename(code_path)],
            cwd=os.path.dirname(code_path),
            timeout=timeout,
            check=True,
        )
    except Exception as e:
        print(f"[tavily_processing] 运行生成代码失败: {e}")


def process_tavily_payload(
    payload: Dict[str, Any], base_dir: str
) -> Tuple[str, List[Tuple[str, str]], List[str], List[str]]:
    """
    对 Tavily 抓取的 payload 进行通用处理：
    - 清洗表格（LLM 生成代码）并保存 cleaned CSV；
    - 基于清洗结果生成数据分析文字；
    - 基于清洗结果生成可视化 PNG 图表。

    :return: (综合分析文本, 新增图表列表[(name, path)], 清洗后CSV路径列表)
    """
    os.makedirs(base_dir, exist_ok=True)
    # 归档：保存 LLM 生成的清洗/可视化代码，便于复现
    code_root = os.path.join(base_dir, "generated_code")
    clean_code_dir = os.path.join(code_root, "clean")
    viz_code_dir = os.path.join(code_root, "viz")
    os.makedirs(clean_code_dir, exist_ok=True)
    os.makedirs(viz_code_dir, exist_ok=True)

    cleaned_csvs: List[str] = []
    raw_csvs: List[str] = []
    charts: List[Tuple[str, str]] = []
    analyses: List[str] = []

    tables = (payload or {}).get("tables") or []
    tables = tables[:3]  # 最多处理前三张表

    for idx, t in enumerate(tables, start=1):
        csv_text = t.get("csv", "")
        if not csv_text or not csv_text.strip():
            continue

        raw_csv_path = os.path.join(base_dir, f"raw_table_{idx:02d}.csv")
        cleaned_csv_path = os.path.join(base_dir, f"cleaned_table_{idx:02d}.csv")
        clean_code_path = os.path.join(clean_code_dir, f"clean_code_{idx:02d}.py")
        viz_code_path = os.path.join(viz_code_dir, f"viz_code_{idx:02d}.py")
        chart_dir = os.path.join(base_dir, f"charts_{idx:02d}")
        os.makedirs(chart_dir, exist_ok=True)

        # 写原始 CSV
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        raw_csvs.append(raw_csv_path)

        # 元信息
        try:
            table_meta = _build_table_meta_from_csv_text(csv_text)
        except Exception as e:
            print(f"[tavily_processing] 解析表格失败: {e}")
            continue

        # 生成并运行清洗代码
        clean_code = _gen_cleaning_code(table_meta, raw_csv_path, cleaned_csv_path)
        _run_generated_code(clean_code, clean_code_path)

        if not os.path.exists(cleaned_csv_path):
            continue

        cleaned_csvs.append(cleaned_csv_path)

        # 基于清洗后数据生成文字分析
        try:
            analysis_text = _gen_analysis_from_cleaned(cleaned_csv_path, table_meta)
            analyses.append(f"【外部表格{idx}分析】\n" + analysis_text.strip())
        except Exception as e:
            print(f"[tavily_processing] 生成分析文字失败: {e}")

        # 生成可视化代码并运行
        try:
            viz_code = _gen_visualization_code(cleaned_csv_path, chart_dir)
            _run_generated_code(viz_code, viz_code_path)
        except Exception as e:
            print(f"[tavily_processing] 生成可视化代码失败: {e}")

        # 收集图表
        try:
            produced_png = False
            for fname in os.listdir(chart_dir):
                if fname.lower().endswith(".png"):
                    produced_png = True
                    charts.append(
                        (
                            f"外部权威表格{idx}图：{fname}",
                            os.path.join(chart_dir, fname),
                        )
                    )
            # 如果确实生成了图片，把对应的可视化代码同步一份到该图表目录，方便“图-代码”配套查看
            if produced_png and os.path.exists(viz_code_path):
                try:
                    shutil.copyfile(viz_code_path, os.path.join(chart_dir, "_viz_code.py"))
                except Exception:
                    pass
        except Exception as e:
            print(f"[tavily_processing] 收集图表失败: {e}")

    combined_analysis = (
        "\n\n".join(analyses) if analyses else "外部权威表格分析：暂无可用表格或分析结果。"
    )
    return combined_analysis, charts, cleaned_csvs, raw_csvs


