from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.loaders.common import attach_metadata, compute_file_hash

def load(path: Path, docs_dir: Path) -> list[Document]:
    """
    PDF 逐页加载，每页元数据带 page 编号（1-based）。
    页内再做递归字符切分，chunk 继承父页的 page 元数据。
    """
    file_hash = compute_file_hash(path)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    all_chunks: list[Document] = []
    for i, page_doc in enumerate(pages, start=1):
        page_chunks = splitter.split_documents([page_doc])
        page_chunks = attach_metadata(page_chunks, path, docs_dir, file_hash, page=i, section="")
        all_chunks.extend(page_chunks)

    return all_chunks