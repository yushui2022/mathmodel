import math
import re
import asyncio
from typing import List, Tuple

from config import rerank_embeddings, vectorstore

# 全局缓存
_retrieval_cache = {}
_retrieval_lock = asyncio.Lock()


def _keyword_score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    tokens = [t.strip() for t in re.split(r"[ ,，。；;、\n\t]", query) if t.strip()]
    if not tokens:
        return 0.0
    score = 0
    for tok in tokens:
        score += text.count(tok)
    return float(score)


def _hybrid_retrieve(query: str, k: int = 5, initial_k: int = 12):
    """
    混合检索：向量召回 + 关键词匹配 + reranker 重排序
    """
    results = vectorstore.similarity_search_with_score(query, k=initial_k)
    if not results:
        return []
    docs, dist_scores = zip(*results)
    vec_sims = [1.0 / (1.0 + float(d)) for d in dist_scores]

    kw_scores = [_keyword_score(query, d.page_content or "") for d in docs]
    if max(kw_scores) > 0:
        kw_scores = [s / max(kw_scores) for s in kw_scores]
    else:
        kw_scores = [0.0 for _ in kw_scores]

    texts = [d.page_content or "" for d in docs]
    q_emb = rerank_embeddings.embed_query(query)
    doc_embs = rerank_embeddings.embed_documents(texts)

    rerank_sims = []
    for emb in doc_embs:
        dot = sum(a * b for a, b in zip(q_emb, emb))
        q_norm = math.sqrt(sum(a * a for a in q_emb)) or 1.0
        d_norm = math.sqrt(sum(a * a for a in emb)) or 1.0
        rerank_sims.append(dot / (q_norm * d_norm))

    min_r, max_r = min(rerank_sims), max(rerank_sims)
    if max_r > min_r:
        rerank_sims = [(s - min_r) / (max_r - min_r) for s in rerank_sims]
    else:
        rerank_sims = [0.5 for _ in rerank_sims]

    final_scores = []
    for v, k_s, r in zip(vec_sims, kw_scores, rerank_sims):
        score = 0.6 * r + 0.3 * v + 0.1 * k_s
        final_scores.append(score)

    ranked = sorted(zip(docs, final_scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:k]]
    return top_docs


def get_cached_retrieval(query: str):
    if query not in _retrieval_cache:
        docs = _hybrid_retrieve(query, k=5, initial_k=12)
        _retrieval_cache[query] = docs
    return _retrieval_cache[query]


async def get_cached_retrieval_async(query: str):
    async with _retrieval_lock:
        if query in _retrieval_cache:
            return _retrieval_cache[query]
    docs = await asyncio.to_thread(_hybrid_retrieve, query, 5, 12)
    async with _retrieval_lock:
        _retrieval_cache[query] = docs
        return _retrieval_cache[query]

__all__ = ["get_cached_retrieval", "get_cached_retrieval_async"]

