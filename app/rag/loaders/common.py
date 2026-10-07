import hashlib
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document

def compute_file_hash(path: Path) -> str:
    """MD5 文件内容，用于增量索引判断是否变化"""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda : f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def base_metadata(path: Path, docs_dir: Path, file_hash: str, *, page: int = 0, section: str = "") -> dict:
    """
    构造 chunk 元数据。注意 Chroma 的 metadata 只支持 str/int/float/bool，
    所以空值用 0 或 ""，不要用 None / list。
    """
    rel_path = path.resolve().relative_to(docs_dir.resolve()).as_posix()
    return {
        "source": path.name,
        "file_path": rel_path,
        "file_type": path.suffix.lower().lstrip("."),
        "file_hash": file_hash,
        "page": page,
        "section": section,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }

def attach_metadata(docs: list[Document], path: Path, docs_dir: Path, file_hash: str, *, page: int = 0, section: str = "") -> list[Document]:
    """给一批 doc 打上基础元数据"""
    base = base_metadata(path, docs_dir, file_hash, page=page, section=section)
    for d in docs:
        d.metadata = {**d.metadata, **base}
    return docs