from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.rag.loaders.common import attach_metadata, compute_file_hash

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

def load(path: Path, docs_dir: Path) -> list[Document]:
    """
    Markdown 分两阶段：
    1. MarkdownHeaderTextSplitter 按标题层级切，保留 h1/h2/h3 到元数据
    2. RecursiveCharacterTextSplitter 对超长 section 再切
    """
    file_hash = compute_file_hash(path)
    text = path.read_text(encoding="utf-8")

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    sections = md_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    all_chunks: list[Document] = []
    for sec in sections:
        # 拼出 section 字符串，如 "第二章 > 2.3 功率"
        section_str = " > ".join(sec.metadata.get(k, "") for k in ("h1", "h2", "h3") if sec.metadata.get(k))
        sub_chunks = char_splitter.split_documents([sec])
        sub_chunks = attach_metadata(sub_chunks, path, docs_dir, file_hash, page=0, section=section_str)
        all_chunks.extend(sub_chunks)

    return all_chunks