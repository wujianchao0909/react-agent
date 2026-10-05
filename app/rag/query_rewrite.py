from langchain_core.prompts import ChatPromptTemplate

from app.llm import llm

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是查询改写助手，服务对象是本地知识库检索系统。\n"
     "根据用户原始问题，生成 {n} 个**不同表述**的查询，要求：\n"
     "1. 语义与原问题一致；\n"
     "2. 使用不同的关键词和句式，覆盖更多检索路径；\n"
     "3. 每行一个，不要编号、不要解释、不要引号；\n"
     "4. 保留原问题中的专有名词、数字、缩写。"),
    ("human", "原始问题：{query}"),
])

async def rewrite_query(query: str, n: int = 3) -> list[str]:
    """
    返回 [原始 query, 变体1, 变体2, ...]。
    失败时降级为 [原始 query]，保证不阻断主流程。
    """
    try:
        chain = REWRITE_PROMPT | llm
        result = await chain.ainvoke({"query": query, "n": n})
        lines = [
            line.strip().lstrip("-•*· ").strip().strip('"').strip("'")
            for line in result.content.splitlines()
        ]
        variants = [l for l in lines if l and l != query][:n]
        return [query] + variants
    except Exception as e:
        print(f"[query_rewrite] 失败，降级：{e}")
        return [query]