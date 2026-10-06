import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.builder import build_agent_with_memory
from app.graph.checkpointer import build_checkpointer


async def stream_chat(app, query: str, config: dict):
    print(f"\n用户: {query}")
    print("Agent: ", end="", flush=True)

    buffer: list[str] = []

    async for event in app.astream(
        {"messages": HumanMessage(content=query)},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        mode, data = event
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, AIMessage) and chunk.content:
                buffer.append(chunk.content)

        elif mode == "updates":
            for node_name, update in data.items():
                if node_name == "agent":
                    msg = update["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        buffer.clear()
                        for tc in msg.tool_calls:
                            print(f"\n  🔧 调用 {tc['name']}({tc['args']})")
                            print("Agent: ", end="", flush=True)
                    else:
                        for piece in buffer:
                            print(piece, end="", flush=True)
                        buffer.clear()
                elif node_name == "tools":
                    for msg in update["messages"]:
                        name = getattr(msg, "name", "unknown")
                        print(f"  📋 {name} 返回: {msg.content}")
                        print("Agent: ", end="", flush=True)
print()

async def main():
    async with build_checkpointer(use_persistent=True) as checkpointer:
        app = build_agent_with_memory(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "user-002"}}

        while True:
            q = input("\n请输入问题: ").strip()
            if q in ("exit", "quit", "q"):
                break
            if not q:
                continue
            await stream_chat(app, q, config)

if __name__ == "__main__":
    asyncio.run(main())
