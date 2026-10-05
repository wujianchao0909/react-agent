from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings

@asynccontextmanager
async def build_checkpointer(use_persistent: bool = False):
    """
    异步上下文管理器，返回一个 checkpointer。

    - use_persistent=True  → AsyncSqliteSaver，落盘到 settings.sqlite_path
    - use_persistent=False → MemorySaver，进程退出即丢

    用法：
        async with build_checkpointer(True) as cp:
            app = build_agent_with_memory(checkpointer=cp)
            ...
    """
    if use_persistent:
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            yield saver
    else:
        yield MemorySaver()