from langchain_core.messages import SystemMessage

from app.llm import llm
from app.tools import ALL_TOOLS
from app.graph.state import AgentState

SYSTEM_PROMPT = SystemMessage(content="""
你是知识库问答助手，可以调用工具完成任务。

工具使用规则：
1. 当用户询问事实性知识、概念、定义、资料类问题时，优先调用 search_knowledge_real 检索本地知识库；
2. 当用户请求数学计算时，调用 calculator；
3. 当用户询问天气时，调用 get_weather_real；
4. 知识库返回有效内容时，以知识库内容为回答主要依据，可归纳总结；
5. 知识库返回空、报错、查不到资料时，只回复「知识库中未找到相关信息。」，不要使用内置知识作答；
6. 不要猜测，不要编造，宁可拒答也不要说错。
""")


llm_with_tools = llm.bind_tools(ALL_TOOLS)

async def agent_node(state: AgentState) -> dict:
    input_messages = [SYSTEM_PROMPT] + state["messages"]
    response = await llm_with_tools.ainvoke(input_messages)
    if response.tool_calls:
        print(f"\nLLM 返回是否带 tool_calls: {bool(response.tool_calls)}")
    return {"messages": [response]}