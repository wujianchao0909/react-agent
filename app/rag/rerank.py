from langchain_core.documents import Document

from app.config import settings

reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[rerank] 加载模型 {settings.reranker_path} ...")
        reranker = CrossEncoder(settings.reranker_path, max_length=512)
    return reranker

def rerank(query: str, docs: list[Document], top_k: int = 3) -> list[tuple[Document, float]]:
    """
    CrossEncoder 精排：把 (query, doc) 作为一对送入模型打分。
    返回 [(doc, score), ...]，按分数降序。
    """
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(docs, (float(s) for s in scores)),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]