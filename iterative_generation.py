from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

from langchain_core.prompts import ChatPromptTemplate


@dataclass
class LengthSpec:
    """
    用“字符数”做长度控制（对中文最稳定）。
    """

    target: int
    tolerance_ratio: float = 0.10  # ±10%

    @property
    def min_len(self) -> int:
        return int(self.target * (1.0 - self.tolerance_ratio))

    @property
    def max_len(self) -> int:
        return int(self.target * (1.0 + self.tolerance_ratio))


EXPAND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是数学建模论文写作专家。你的任务是对“上一轮草稿”进行扩写补全，使其满足字数与规范。
强约束：
- 必须保留并复用上一轮草稿的结构与标题，不得删减已有内容；
- 只能在原文基础上补充细节、推导、参数、步骤、对比与解释；
- 不得引入与“符号说明”冲突的新符号；如需新增符号，必须同时给出清晰定义并与符号说明一致；
- 数学公式必须使用 LaTeX，且保持原有公式编号不被破坏；新增公式需延续编号风格（如(3.4)、(4.7)）；
- 禁止空话套话与模糊措辞（如“大概”“可能”“若干”）。
输出要求：只输出扩写后的完整正文（不是修改说明），中文。
""",
        ),
        (
            "human",
            """
【目标长度】请将正文扩写到 {min_len}~{max_len} 字符（当前约 {current_len} 字符）。

【必须满足的规范摘要】
{requirements}

【可用上下文（用于补充细节与引用）】
{context_blob}

【上一轮草稿】
{draft}
""",
        ),
    ]
)


def expand_until_length(
    llm,
    draft: str,
    length_spec: LengthSpec,
    requirements: str,
    context_blob: str,
    max_rounds: int = 4,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """
    多轮扩写：如果 draft 长度不足，则循环调用 LLM 扩写到目标区间。
    - 用字符长度粗控（对中文更稳定）
    - 超过 max_rounds 则返回当前版本
    """
    text = (draft or "").strip()
    if not text:
        return text

    for r in range(max_rounds):
        cur = len(text)
        if length_spec.min_len <= cur <= length_spec.max_len:
            return text

        if progress_cb:
            progress_cb(f"扩写回合 {r+1}/{max_rounds}：当前{cur}字，目标{length_spec.min_len}-{length_spec.max_len}字")

        msgs = EXPAND_PROMPT.format_messages(
            min_len=length_spec.min_len,
            max_len=length_spec.max_len,
            current_len=cur,
            requirements=requirements,
            context_blob=context_blob,
            draft=text,
        )
        resp = llm.invoke(msgs)
        text = getattr(resp, "content", str(resp)).strip()

    return text








