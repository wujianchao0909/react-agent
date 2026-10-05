import os
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

embedding = None

def get_embedding() -> HuggingFaceEmbeddings:
    """加载本地 embedding 模型"""
    global embedding

    if embedding is not None:
        return embedding

    if not os.path.isdir(settings.embedding_path):
        raise FileNotFoundError(
            f"本地模型目录不存在：{settings.embedding_path}\n"
            f"请先执行：hf download (replaced by your model name)"
            f"--local-dir {settings.embedding_path}"
        )

    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_path,          # 指向本地目录
        model_kwargs={"device": "cpu"},           # 有 GPU 就改 "cuda"
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(os.path.join(settings.embedding_path, "_cache")),
    )

    return embedding