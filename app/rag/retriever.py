import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sqlalchemy.testing.suite.test_reflection import metadata

from app.rag.vectorstore import get_vectorstore

hybrid_retriever = None

def tokenize(text: str) -> list[str]:
    """中文 BM25 分词：jieba 搜索引擎模式，兼顾长词与短词"""
    return[t for t in jieba.cut_for_search(text) if t.strip()]

def get_hybrid_retriever() -> EnsembleRetriever:
    """
    BM25（关键词）+ 向量（语义）混合检索。
    - BM25 权重 0.4：擅长专有名词、缩写、精确数字
    - 向量权重 0.6：擅长同义改写、语义泛化
    """
    global hybrid_retriever
    if hybrid_retriever is not None:
        return hybrid_retriever

    vs = get_vectorstore()

    data = vs.get()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    if not docs:
        raise ValueError("向量库为空，无法构建 BM25 索引")

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    bm25.k = 10

    vector = vs.as_retriever(search_kwargs={"k": 10})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6],
    )

    return hybrid_retriever