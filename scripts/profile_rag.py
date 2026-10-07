"""分阶段统计 RAG 链路耗时"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from app.rag.retriever import get_hybrid_retriever
from app.rag.query_rewrite import rewrite_query
from app.rag.rerank import rerank


async def main():
    query = "电压是什么"

    # 阶段 1：改写
    t = time.perf_counter()
    queries = await rewrite_query(query, n=3)
    print(f"[1] Query 改写: {time.perf_counter()-t:.2f}s  变体数={len(queries)}")
    for q in queries:
        print(f"    {q}")

    # 阶段 2：混合检索（串行）
    t = time.perf_counter()
    retriever = get_hybrid_retriever()
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)
    t_serial = time.perf_counter() - t
    print(f"[2] 混合检索（串行 {len(queries)} 路）: {t_serial:.2f}s  总召回={len(all_docs)}")

    # 阶段 3：Rerank
    seen = set()
    candidates = []
    for d in all_docs:
        k = d.page_content[:80]
        if k not in seen:
            seen.add(k)
            candidates.append(d)
    t = time.perf_counter()
    ranked = rerank(query, candidates, 3)
    print(f"[3] Rerank（{len(candidates)} 候选 -> top3）: {time.perf_counter()-t:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())