from langchain_core.messages import SystemMessage

from app.llm import llm
from app.tools import ALL_TOOLS
from app.graph.state import AgentState

SYSTEM_PROMPT = SystemMessage(content="""
你是知识库问答助手。

**工具选择（按意图分流）**：
1. 数学计算（加减乘除、开方、幂、三角函数）→ 调用 calculator
2. 天气查询 → 调用 get_weather_real
3. **其余所有问题**（概念、事实、知识、资料、闲聊、常识）→ **必须调用 search_knowledge_real**

**强制规则**：
- 涉及知识/事实类的问题，**不管你知不知道答案、不管历史对话里是否已经查过**，每次都要**重新调用** search_knowledge_real；
- 不要根据历史对话推断「知识库里有什么、没什么」；
- 工具返回有效内容时，基于知识库内容回答，可归纳总结；
- 工具返回「知识库中未找到相关信息。」时，**只回复这一句**，不要用内置知识补充，不要编造。
""")


llm_with_tools = llm.bind_tools(ALL_TOOLS)

async def agent_node(state: AgentState) -> dict:
    input_messages = [SYSTEM_PROMPT] + state["messages"]
    response = await llm_with_tools.ainvoke(input_messages)
    if response.tool_calls:
        print(f"\nLLM 返回是否带 tool_calls: {bool(response.tool_calls)}")
    return {"messages": [response]}