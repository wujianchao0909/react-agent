import torch
from langchain_core.documents import Document
from app.config import settings

reranker = None


def resolve_device() -> str:
    if settings.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.device


def get_reranker():
    global reranker
    if reranker is None:
        from sentence_transformers import CrossEncoder
        device = resolve_device()
        print(f"[rerank] loading {settings.reranker_path} on {device}")
        reranker = CrossEncoder(
            settings.reranker_path,
            max_length=512,
            device=device,
        )
    return reranker


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 3,
) -> list[tuple[Document, float]]:
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(docs, (float(s) for s in scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]