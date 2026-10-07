from pathlib import Path
from langchain_core.documents import Document

from app.config import settings
from app.rag.loaders import text as text_loader
from app.rag.loaders import pdf as pdf_loader
from app.rag.loaders import docx as docx_loader
from app.rag.loaders import markdown as markdown_loader
from app.rag.loaders import html as html_loader

LOADERS = {
    ".txt": text_loader.load,
    ".pdf": pdf_loader.load,
    ".docx": docx_loader.load,
    ".md": markdown_loader.load,
    ".markdown": markdown_loader.load,
    ".html": html_loader.load,
    ".htm": html_loader.load,
}

def is_supported(path: Path) -> bool:
    return path.suffix.lower() in LOADERS

def load_and_split(path: Path, docs_dir: Path | None = None) -> list[Document]:
    """
    统一入口：根据扩展名分发到具体 loader。
    docs_dir 默认 settings.docs_dir，用于计算相对路径（元数据里的 file_path）。
    """
    if docs_dir is None:
        docs_dir = Path(settings.docs_dir)

    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"不支持的文件类型：{ext}")

    return LOADERS[ext](path, docs_dir)