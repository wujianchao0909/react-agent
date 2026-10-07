from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loaders.common import attach_metadata, compute_file_hash

def load(path: Path, docs_dir: Path) -> list[Document]:
    """
    HTML：BS4 提取正文，把 h1~h3 作为 section 边界，其他内容按段落切。
    """
    file_hash = compute_file_hash(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # 去掉 script / style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # 按标题切分（类似 markdown）
    blocks: list[tuple[str, str]] = []   # (section, text)
    current_section= ""
    buffer: list[str] = []

    def flush():
        nonlocal buffer
        text = "\n".join(t for t in buffer if t.strip())
        if text.strip():
            blocks.append((current_section, text))
        buffer = []

    for elem in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "td"]):
        if elem.name in ("h1", "h2", "h3"):
            flush()
            current_section = elem.get_text(strip=True)
        else:
            t = elem.get_text(" ", strip=True)
            if t:
                buffer.append(t)
    flush()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    all_chunks: list[Document] = []
    for section, text in blocks:
        doc = Document(page_content=text)
        sub = splitter.split_documents([doc])
        sub = attach_metadata(sub, path, docs_dir, file_hash, page=0, section=section)
        all_chunks.extend(sub)

    return all_chunks