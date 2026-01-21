"""
Tavily 联网检索 + 权威表格抓取 + 本地缓存（SQLite）

目标：
- 用 Tavily 搜索定位权威来源页面
- 自动抓取页面中的表格（pandas.read_html）
- 将结果缓存到本地 SQLite，避免重复联网
- 输出可直接写入论文的“数据摘要 + 引用列表”
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import io
from pathlib import Path


@dataclass
class SourceItem:
    title: str
    url: str
    snippet: str = ""

# 固定 Key（用户要求写死），若同时设置了环境变量 TAVILY_API_KEY，则环境变量优先。
DEFAULT_TAVILY_API_KEY = "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"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


class WebDataCache:
    def __init__(self, db_path: str = "web_cache.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_cache (
                    key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT payload FROM web_cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "REPLACE INTO web_cache(key, created_at, payload) VALUES (?, ?, ?)",
                (key, time.time(), json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()


class TavilyClient:
    """
    轻量 Tavily REST Client（避免额外 SDK 依赖；如你安装了 tavily-python 也不冲突）
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        # 优先使用显式传入，其次环境变量，最后使用写死的默认 key
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "") or DEFAULT_TAVILY_API_KEY
        self.timeout = timeout
        self.base_url = "https://api.tavily.com/search"

    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        if not self.enabled():
            return []
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            # 更偏“权威数据源”的搜索意图（tavily 会尽力给高质量来源）
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) or []
        items: List[SourceItem] = []
        for r in results:
            items.append(
                SourceItem(
                    title=r.get("title", "") or "",
                    url=r.get("url", "") or "",
                    snippet=r.get("content", "") or "",
                )
            )
        return items


def extract_tables_from_url(url: str, max_tables: int = 3) -> List[pd.DataFrame]:
    """
    抓取网页中的表格（read_html 对多数权威站点/报告的表格很有效）
    """
    # 使用 pandas 直接解析（内部会用 lxml/bs4）
    tables = pd.read_html(url, flavor="lxml")
    return tables[:max_tables]


def summarize_tables(tables: List[pd.DataFrame], max_rows: int = 5) -> str:
    parts: List[str] = []
    for i, df in enumerate(tables, 1):
        df2 = df.copy()
        # 统一列名，避免过宽
        df2.columns = [str(c)[:40] for c in df2.columns]
        parts.append(f"表{i}（{df2.shape[0]}行×{df2.shape[1]}列）示例：\n{df2.head(max_rows).to_string(index=False)}")
    return "\n\n".join(parts)


def _df_preview(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 12) -> Dict[str, Any]:
    """
    给前端用的表格预览（限制行列数，避免返回过大）。
    """
    df2 = df.copy()
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    df2.columns = [str(c) for c in df2.columns]
    rows = df2.astype(str).replace({pd.NA: ""}).values.tolist()
    return {"columns": list(df2.columns), "rows": rows, "shape": [int(df.shape[0]), int(df.shape[1])]}


def _save_payload_to_run_dir(payload: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    """
    将本次爬取结果落盘到 run_dir：
    - payload.json
    - tables/*.csv
    同时把每个 table 增加 saved_path（相对路径）字段，便于前端下载。
    """
    run_path = Path(run_dir)
    tables_path = run_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

    # 保存表格 CSV
    new_tables: List[Dict[str, Any]] = []
    for i, t in enumerate(payload.get("tables", []) or []):
        csv_text = t.get("csv", "")
        fname = f"table_{i+1:03d}.csv"
        fpath = tables_path / fname
        try:
            fpath.write_text(csv_text, encoding="utf-8")
            saved_rel = str(Path(run_path.name) / "tables" / fname).replace("\\", "/")
        except Exception:
            saved_rel = ""
        t2 = dict(t)
        t2["saved_path"] = saved_rel
        new_tables.append(t2)

    payload2 = dict(payload)
    payload2["tables"] = new_tables
    payload2["raw_csvs"] = [t.get("saved_path", "") for t in new_tables if t.get("saved_path")]

    # 保存 payload.json（包含 sources + summary + tables 元信息）
    (run_path / "payload.json").write_text(json.dumps(payload2, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload2


def fetch_authoritative_tables_with_cache(
    query: str,
    cache: WebDataCache,
    tavily: TavilyClient,
    max_sources: int = 3,
    max_tables_per_source: int = 2,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    返回结构：
    {
      "query": str,
      "sources": [{"title":..., "url":..., "snippet":...}, ...],
      "tables": [{"source_url":..., "table_idx":..., "csv":...}, ...],
      "summary": str,
    }
    """
    cache_key = "tavily_tables:" + _sha1(query)
    cached = cache.get(cache_key)
    if cached:
        # 若指定了 run_dir，也落一份到本次运行目录，方便前端展示/下载
        if run_dir:
            try:
                return _save_payload_to_run_dir(cached, run_dir)
            except Exception:
                return cached
        return cached

    sources = tavily.search(query, max_results=max_sources)
    all_tables: List[Dict[str, Any]] = []

    for src in sources:
        try:
            tables = extract_tables_from_url(src.url, max_tables=max_tables_per_source)
            for ti, df in enumerate(tables):
                all_tables.append(
                    {
                        "source_url": src.url,
                        "table_idx": ti,
                        "csv": df.to_csv(index=False),
                        "preview": _df_preview(df),
                    }
                )
        except Exception:
            # 某些站点可能禁止 read_html 或没有表格，跳过即可
            continue

    # 生成摘要（只展示前几个表格）
    preview_tables: List[pd.DataFrame] = []
    for t in all_tables[: min(3, len(all_tables))]:
        try:
            preview_tables.append(pd.read_csv(io.StringIO(t["csv"])))
        except Exception:
            continue

    summary = summarize_tables(preview_tables) if preview_tables else "未抓取到可解析的权威表格（可能页面无表格或被反爬）。"

    payload = {
        "query": query,
        "sources": [src.__dict__ for src in sources],
        "tables": all_tables,
        "summary": summary,
    }
    cache.set(cache_key, payload)
    if run_dir:
        try:
            payload = _save_payload_to_run_dir(payload, run_dir)
        except Exception:
            pass
    return payload


