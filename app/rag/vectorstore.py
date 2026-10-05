import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from app.config import settings
from app.rag.embedding import get_embedding

vectorstore = None

def get_vectorstore() -> Chroma:
    global vectorstore
    if vectorstore is not None:
        return vectorstore

    embedding = get_embedding()

    if os.path.exists(settings.chroma_dir):
        vectorstore = Chroma(persist_directory=settings.chroma_dir,
                             embedding_function=embedding
                             )
        return vectorstore

    if not os.path.exists(settings.chroma_dir):
        raise FileNotFoundError(f"知识库目录不存在：{settings.docs_dir}")

    documents = []

    for fname in os.listdir(settings.docs_dir):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(settings.docs_dir, fname), encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        raise ValueError("知识库中没有可加载的 txt 文档")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding_function=embedding,
                                        persist_directory=settings.chroma_dir
                                        )

    return vectorstore