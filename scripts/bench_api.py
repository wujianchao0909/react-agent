"""通过 HTTP 接口测容器内 API 的真实延迟"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

API = "http://localhost:8000/api/chat/stream"

QUERIES = [
    ("电压是什么？", "rag_hit"),
    ("什么叫电势差？", "rag_hit"),
    ("请介绍一下量子纠缠", "rag_miss"),
    ("北京天气如何？", "tool_weather"),
    ("计算 (100-20)*3", "tool_calc"),
]


def run_one(client: httpx.Client, query: str, thread_id: str) -> float:
    payload = {"message": query, "thread_id": thread_id}
    t = time.perf_counter()
    with client.stream("POST", API, json=payload) as resp:
        for _ in resp.iter_lines():
            pass
    return (time.perf_counter() - t) * 1000


def main():
    print(f"Benchmarking {API}\n")
    with httpx.Client(timeout=None) as client:
        total = 0.0
        for i, (query, tag) in enumerate(QUERIES):
            ms = run_one(client, query, f"bench-{i}")
            total += ms
            print(f"  [{tag:12s}] {ms:7.0f} ms   {query}")

        print(f"\n平均延迟：{total / len(QUERIES):.0f} ms")


if __name__ == "__main__":
    main()