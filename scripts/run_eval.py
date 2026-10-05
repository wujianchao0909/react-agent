import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.rerank import get_reranker
from app.graph.builder import build_agent_with_memory
from app.graph.checkpointer import build_checkpointer
from eval.dataset import load_dataset
from eval.runner import run_all
from eval.metrics import aggregate

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

async def main():
    print("预热 reranker...")
    await asyncio.to_thread(get_reranker)
    cases = load_dataset()
    print(f"加载评测集：{len(cases)} 条\n")

    REPORT_DIR.mkdir(exist_ok=True)

    async with build_checkpointer(use_persistent=False) as cp:
        agent = build_agent_with_memory(checkpointer=cp)
        results = await run_all(agent=agent, cases=cases)

    summary = aggregate(results)

    print("\n" + "=" * 60)
    print(f"总用例        : {summary.total}")
    print(f"通过率        : {summary.pass_rate:.1%}  ({summary.passed}/{summary.total})")
    print(f"工具调用准确率: {summary.tool_accuracy:.1%}")
    print(f"关键词命中率  : {summary.keyword_accuracy:.1%}")
    print(f"拒答正确率    : {summary.refusal_correct / summary.total:.1%}")
    print(f"平均延迟      : {summary.avg_latency_ms:.0f} ms")
    print(f"总 token      : in={summary.total_input_tokens} out={summary.total_output_tokens}")
    print("\n按类别：")
    for cat, stat in summary.by_category.items():
        rate = stat["passed"] / stat["total"] if stat["total"] else 0
        print(f"  {cat:16s}: {stat['passed']}/{stat['total']}  ({rate:.0%})")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": summary.total,
            "passed": summary.passed,
            "pass_rate": summary.pass_rate,
            "tool_accuracy": summary.tool_accuracy,
            "keyword_accuracy": summary.keyword_accuracy,
            "avg_latency_ms": summary.avg_latency_ms,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "by_category": summary.by_category,
        },
        "cases": [
            {
                "id": r.id, "category": r.category, "query": r.query,
                "answer": r.answer[:200],
                "expected_tools": r.expected_tools,
                "actual_tools": r.actual_tools,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
    }
    out = REPORT_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存：{out}")

if __name__ == "__main__":
    asyncio.run(main())