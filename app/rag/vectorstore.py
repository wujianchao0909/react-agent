from pathlib import Path
from langchain_chroma import Chroma

from app.config import settings
from app.rag.embedding import get_embedding

vectorstore = None

def get_vectorstore() -> Chroma:
    """
    返回全局唯一的 Chroma 实例。
    文档加载/切分/灌库交给 indexer，这里只保证连接可用。
    """
    global vectorstore
    if vectorstore is not None:
        return vectorstore

    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embedding(),
        collection_name="knowledge"
    )

    return vectorstore