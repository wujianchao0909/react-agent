import asyncio
import os
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from app.rag.retriever import get_hybrid_retriever
from app.rag.query_rewrite import rewrite_query
from app.rag.rerank import rerank
from app.llm import llm

FINAL_K = 3
RELATIVE_MARGIN = 0.05

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "判断以下文档片段是否与用户问题相关。\n"
     "只输出 yes 或 no，不要解释。\n"
     "如果文档只是字面相似但语义无关（如都是「天气」但一个问北京一个问上海），输出 no。"),
    ("human", "问题：{query}\n\n文档：{doc}"),
])

async def is_relevant(query: str, doc: str) -> bool:
    result = await (JUDGE_PROMPT | llm).ainvoke({"query": query, "doc": doc[:500]})
    return "yes" in result.content.lower()

@tool
async def search_knowledge_real(query: Annotated[str, Field(description="搜索关键词或问题")]) -> str:
    """在本地知识库中搜索相关信息。当用户询问需要查阅资料的事实时使用。"""
    queries = await rewrite_query(query, n=2)

    retriever = get_hybrid_retriever()
    seen: set[str] = set()
    candidates = []

    all_docs = await asyncio.gather(*[
        asyncio.to_thread(retriever.invoke, q) for q in queries
    ])
    for docs in all_docs:
        for d in docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                candidates.append(d)

    if not candidates:
        return "知识库中未找到相关信息。"

    ranked = await asyncio.to_thread(rerank, query, candidates, FINAL_K)

    if not ranked:
        return "知识库中未找到相关信息。"

    top_score = ranked[0][1]

    kept = []
    for doc, score in ranked:
        if await is_relevant(query, doc.page_content):
            kept.append((doc, score))
    if not kept:
        return "知识库中未找到相关信息。"

    print(f"  [rerank] top scores: {[f'{s:.4f}' for _, s in ranked]}")

    output = []
    for doc, score in kept:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        output.append(f"[来源:{source}，rerank分数:{score:.4f}]\n{doc.page_content}")


    return "\n\n---\n\n".join(output)