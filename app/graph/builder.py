from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState
from app.graph.nodes import agent_node
from app.tools import ALL_TOOLS

def handle_tool_error(exc: Exception):
    """
    工具执行异常兜底：转成字符串作为 ToolMessage 返回。
    绝不 raise，避免留下「孤儿 tool_calls」污染 checkpoint。
    """
    return f"工具执行失败（{type(exc).__name__}）：{exc}"

def should_retry(exc: Exception) -> bool:
    """
    保留给 RetryPolicy：只有工具内部不吞异常时才会被触发。
    目前工具内部大多已 try/except，这里主要是兜底。
    """
    retryable = (TimeoutError, ConnectionResetError, OSError)
    non_retryable = (ValueError, TypeError, ImportError, FileNotFoundError)
    if isinstance(exc, retryable):
        return True
    if isinstance(exc, non_retryable):
        return False
    return True

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=handle_tool_error),
                   retry_policy=RetryPolicy(
                       max_attempts=3,
                       initial_interval=0.5,
                       backoff_factor=2.0,
                       retry_on=should_retry
                   ))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools","agent")
    return graph

def build_agent_with_memory(checkpointer=None):
    """
    构建带记忆的 Agent。
    checkpointer 由调用方负责创建与释放（异步场景务必用 AsyncSqliteSaver）。
    不传则默认用内存 MemorySaver。
    """
    graph = build_agent()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
