import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

embedding = None


def _resolve_device() -> str:
    if settings.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.device


def get_embedding() -> HuggingFaceEmbeddings:
    global embedding
    if embedding is not None:
        return embedding

    if not os.path.isdir(settings.embedding_path):
        raise FileNotFoundError(
            f"本地模型目录不存在：{settings.embedding_path}"
        )

    device = _resolve_device()
    print(f"[embedding] using device: {device}")

    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=os.path.join(settings.embedding_path, "_cache"),
    )
    return embedding