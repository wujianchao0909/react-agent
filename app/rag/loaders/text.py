from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.loaders.common import attach_metadata, compute_file_hash

def load(path: Path, docs_dir: Path) -> list[Document]:
    file_hash = compute_file_hash(path)
    loader = TextLoader(str(path), encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    return attach_metadata(chunks, path, docs_dir, file_hash)