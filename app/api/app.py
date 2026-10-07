import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.graph.builder import build_agent_with_memory
from app.graph.checkpointer import build_checkpointer
from app.api.routes import chat, health
from app.rag.rerank import get_reranker
from app.rag.indexer import sync_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动时：
      - 打开 AsyncSqliteSaver
      - 编译带记忆的 Agent
      - 存进 app.state
    关闭时：
      - 自动关闭 checkpointer 连接
    """
    async with build_checkpointer(use_persistent=True) as checkpointer:
        # 启动时同步知识库（增量，只有变化的文件才重新索引）
        print("[startup] Syncing knowledge base...")
        await asyncio.to_thread(sync_index)
        print("[startup] Knowledge base synced")

        app.state.agent = build_agent_with_memory(checkpointer=checkpointer)
        
        print("[startup] Agent ready, warming up reranker...")
        await asyncio.to_thread(get_reranker)
        print("[startup] Reranker warmed up")
        print(f"[startup] Agent ready, model={settings.deepseek_model}")
        yield
    print("[shutdown] Agent closed")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge-Base Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api")
    return app

app = create_app()