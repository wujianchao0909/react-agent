import json
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.api.schemas import ChatRequest
from app.api.deps import get_agent

router = APIRouter()

def sse(event: str, data: dict) -> dict:
    """封装 SSE 消息，data 里固定带 type 字段，前端按 type 分发"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}

async def event_generator(agent, req: ChatRequest):
    """把 LangGraph astream 事件翻译成 SSE 事件"""
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        buffer = []
        async for mode, data in agent.astream(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, _meta = data
                if isinstance(chunk, AIMessage) and chunk.content:
                    buffer.append(chunk.content)

            elif mode == "updates":
                for node, update in data.items():
                    if node == "agent":
                        msg = update["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            buffer.clear()
                            for tc in msg.tool_calls:
                                yield sse("tool_calls",
                                          {
                                              "id": tc["id"],
                                              "name": tc["name"],
                                              "args": tc["args"],
                                          })
                        else:
                            for piece in buffer:
                                yield sse("token", {"content": piece})
                            buffer.clear()
                    elif node == "tools":
                        for msg in update["messages"]:
                            yield sse("tool_results", {
                                "name": getattr(msg, "name", "unknown"),
                                "content": str(msg.content),
                            })

        yield sse("done", {})

    except Exception as e:
        yield sse("error", {"message": f"{type(e).__name__}: {e}"})

@router.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest, agent=Depends(get_agent)):
    """
    SSE 流式聊天。

    前端用 EventSource 或 fetch + ReadableStream 消费。
    事件类型：token / tool_call / tool_result / done / error
    """
    return EventSourceResponse(event_generator(agent, req))