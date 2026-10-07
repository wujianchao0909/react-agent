"""诊断 RAG 检索管道各环节"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.vectorstore import get_vectorstore
from app.rag.retriever import get_hybrid_retriever
from app.rag.rerank import rerank


def main():
    query = "红烧肉怎么做"
    print(f"\n{'='*60}\n诊断 query: {query}\n{'='*60}\n")

    vs = get_vectorstore()

    # 1) Chroma 里 food.pdf 有多少 chunk
    result = vs.get(where={"file_path": "food.pdf"})
    ids = result.get("ids", [])
    print(f"[1] food.pdf chunk 总数: {len(ids)}")

    # 2) Chroma 里含「红烧肉」的 chunk
    all_food = vs.get(where={"file_path": "food.pdf"}, include=["documents"])
    docs = all_food.get("documents", [])
    hit_count = sum(1 for d in docs if "红烧肉" in d)
    print(f"[2] food.pdf 中含「红烧肉」的 chunk: {hit_count}")
    if hit_count > 0:
        for d in docs:
            if "红烧肉" in d:
                print(f"    → {d[:120]}...")
                break

    # 3) 纯向量检索 top-5
    print(f"\n[3] 纯向量检索 top-5：")
    vector_docs = vs.similarity_search(query, k=5)
    for i, d in enumerate(vector_docs, 1):
        src = d.metadata.get("file_path", "?")
        page = d.metadata.get("page", 0)
        print(f"    #{i} [{src}|p{page}] {d.page_content[:80]}...")

    # 4) 混合检索 top-10
    print(f"\n[4] 混合检索(BM25+向量) top-10：")
    retriever = get_hybrid_retriever()
    hybrid_docs = retriever.invoke(query)
    for i, d in enumerate(hybrid_docs[:10], 1):
        src = d.metadata.get("file_path", "?")
        page = d.metadata.get("page", 0)
        print(f"    #{i} [{src}|p{page}] {d.page_content[:80]}...")

    # 5) Rerank 分数
    print(f"\n[5] Rerank 分数（对混合检索结果取 top-5）：")
    ranked = rerank(query, hybrid_docs, top_k=5)
    for d, score in ranked:
        src = d.metadata.get("file_path", "?")
        page = d.metadata.get("page", 0)
        print(f"    {score:+.4f} [{src}|p{page}] {d.page_content[:80]}...")


if __name__ == "__main__":
    main()