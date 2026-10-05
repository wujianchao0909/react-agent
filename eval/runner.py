import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from eval.dataset import EvalCase
from eval.metrics import CaseResult

async def run_one(agent, case: EvalCase, thread_id: str) -> CaseResult:
    """
    单条用例：走完整 agent 流程，收集回答、工具调用、延迟、token。
    """
    config = {"configurable": {"thread_id": thread_id}}

    actual_tools: list[str] = []
    answer_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    buffer: list[str] = []

    start = time.perf_counter()

    async for mode, data in agent.astream(
            {"messages": [HumanMessage(content=case["query"])]},
            config=config,
            stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, AIMessage):
                if chunk.content:
                    buffer.append(chunk.content)

                usage = getattr(chunk, "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)

        elif mode == "updates":
            for node, update in data.items():
                if node == "agent":
                    msg = update["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        buffer.clear()
                        for tc in msg.tool_calls:
                            actual_tools.append(tc["name"])
                    else:
                        answer_parts.extend(buffer)
                        buffer.clear()

    latency_ms = (time.perf_counter() - start) * 1000

    return CaseResult(
        id=case["id"],
        category=case["category"],
        query=case["query"],
        answer="".join(answer_parts),
        actual_tools=actual_tools,
        expected_tools=case["expected_tools"],
        expected_keywords=case["expected_keywords"],
        should_refuse=case["should_refuse"],
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

async def run_all(agent, cases: list[EvalCase]) -> list[CaseResult]:
    """
    每条用例独立 thread_id，避免互相污染记忆。
    """
    results = []
    for case in cases:
        thread_id = f"eval-{case['id']}"
        try:
            result = await run_one(agent, case, thread_id)
        except Exception as e:
            print(f"[ERROR] {case['id']}: {e}")
            result = CaseResult(
                id=case["id"], category=case["category"], query=case["query"],
                answer=f"<error: {e}>",
                actual_tools=[], expected_tools=case["expected_tools"],
                expected_keywords=case["expected_keywords"],
                should_refuse=case["should_refuse"],
                latency_ms=0.0,
            )
        mark = "✅" if result.passed else "❌"
        print(f"{mark} [{case['id']}] {case['query'][:30]}  ({result.latency_ms:.0f}ms)")
        results.append(result)

    return results