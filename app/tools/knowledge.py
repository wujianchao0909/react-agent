import asyncio
import os
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

from app.rag.retriever import get_hybrid_retriever
from app.rag.query_rewrite import rewrite_query
from app.rag.rerank import rerank

FINAL_K = 2
MIN_SCORE = 0.0
RELATIVE_MARGIN = 3.0


@tool
async def search_knowledge_real(query: Annotated[str, Field(description="搜索关键词或问题")]) -> str:
    """在本地知识库中搜索相关信息。当用户询问需要查阅资料的事实时使用。"""
    queries = await rewrite_query(query, n=2)

    retriever = get_hybrid_retriever()
    # 多路并发检索
    all_results = await asyncio.gather(*[
        asyncio.to_thread(retriever.invoke, q) for q in queries
    ])
    seen: set[str] = set()
    candidates = []
    for docs in all_results:
        for d in docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                candidates.append(d)

    if not candidates:
        return "知识库中未找到相关信息。"

    candidates = candidates[:12]
    ranked = await asyncio.to_thread(rerank, query, candidates, FINAL_K)

    if not ranked:
        return "知识库中未找到相关信息。"

    top_score = ranked[0][1]
    print(f"  [rerank] top scores: {[f'{s:.4f}' for _, s in ranked]}")

    if top_score < MIN_SCORE:
        return "知识库中未找到相关信息。"

    kept = [(d, s) for d, s in ranked if s >= top_score - RELATIVE_MARGIN]

    output = []
    for doc, score in kept:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0)
        section = doc.metadata.get("section", "")
        loc = f"p{page}" if page else ""
        if section:
            loc = f"{loc} {section}".strip() if loc else section
        header = f"[来源:{source}"
        if loc:
            header += f" {loc}"
        header += f"，rerank分数:{score:.4f}]"
        output.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)