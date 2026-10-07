# Knowledge-Base Q&A Agent

基于 **LangGraph** 的知识库问答 Agent，支持 **RAG + 工具调用 + 多轮记忆 + 流式输出**。

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://langchain-ai.github.io/langgraph/)

---

## 效果

| 指标 | 结果 |
|---|---|
| 任务通过率 | **100%** (9/9) |
| 工具调用准确率 | **100%** |
| 关键词命中率 | **100%** |
| 拒答正确率 | **100%** |
| 平均延迟 | **2.44s**（GPU）/ 5.04s（CPU） |
| RAG 单条延迟 | **2.4~3.5s**（GPU）/ 9~14s（CPU） |
| Rerank 加速比 | **~4x**（CPU 14s → GPU 0.3~0.5s） |

> 评测集覆盖 6 类场景。GPU 版本启用 CUDA 12.1 + PyTorch 2.5.1 + Qwen3-Reranker on GPU，RAG 链路端到端延迟相比 CPU 版本下降约 **50%**。

## GPU 加速

本项目支持 CPU / GPU 双模式运行，通过 `DEVICE` 环境变量控制：

| 模式 | 命令 | 延迟 |
|---|---|---|
| **GPU（推荐）** | `DEVICE=auto docker compose up -d` | ~2.4s |
| CPU | `DEVICE=cpu docker compose up -d` | ~5.0s |

GPU 模式依赖：
- NVIDIA GPU（≥ 6GB 显存）
- NVIDIA Container Toolkit（Docker Desktop 已内置）
- CUDA 12.1 驱动

**GPU 优化的技术路径**：
- 基础镜像：`nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- 通过 `docker-compose.yml` 的 `deploy.resources.reservations.devices` 将 GPU 透传给容器
- `torch==2.5.1+cu121` + `transformers==4.51.0`（识别 Qwen3 的最低版本）
- 代码层 `device="auto"` 自动检测 CUDA，本地 CPU 环境自动回退

---

## 架构

```
                        ┌──────────────────────────────────┐
                        │        FastAPI + SSE             │
                        │  POST /api/chat/stream           │
                        │  GET  /health                    │
                        └────────────────┬─────────────────┘
                                         │
                        ┌────────────────▼─────────────────┐
                        │      LangGraph Agent (ReAct)     │
                        │                                  │
                        │    ┌─────────┐    ┌──────────┐   │
                        │    │  agent  │───▶│  tools   │   │
                        │    └────┬────┘    └────┬─────┘   │
                        │         │   loop ◀────┘          │
                        │         ▼                        │
                        │   AsyncSqliteSaver (memory)      │
                        └────┬──────────────┬──────────────┘
                             │              │
                ┌────────────▼───┐    ┌─────▼──────────────┐
                │  Tools         │    │  RAG Pipeline      │
                │  ┌───────────┐ │    │                    │
                │  │ calculator│ │    │ 1. Query 改写(LLM) │
                │  │ weather   │ │    │ 2. BM25+向量 混合  │
                │  │ knowledge │─┼───▶│ 3. CrossEncoder精排│
                │  └───────────┘ │    │ 4. 阈值过滤拒答    │
                └────────────────┘    └────────────────────┘
```

---

## 技术亮点

- **Agent 编排**：LangGraph `StateGraph` + `tools_condition` 条件路由 + `RetryPolicy` 异常重试
- **异步持久化**：`AsyncSqliteSaver` + 全链路异步（`httpx` / `asyncio.to_thread`）
- **RAG 四级流水线**：Query 改写（LLM）→ BM25 + 向量混合召回 → Qwen3-Reranker 精排 → 分数过滤拒答
- **安全工具**：AST 白名单替代 `eval`，杜绝任意代码执行
- **流式输出**：FastAPI + SSE，5 类事件协议（token / tool_call / tool_result / done / error）
- **评测驱动**：JSONL 评测集 + 4 项核心指标 + JSON 报告落盘，支持回归对比
- **GPU 加速**：Docker 容器透传 NVIDIA GPU，Embedding + Reranker 双模型跑在 CUDA 上，Rerank 阶段从 14s 降至亚秒级，RAG 链路端到端延迟下降约 50%

---

## 快速开始

### 方式一：Docker（推荐）

前置：Docker Desktop，本地模型放到项目根：
- `./Qwen3-Embedding-0.6B`
- `./Qwen3-Reranker-0.6B`

```bash
git clone https://github.com/wujianchao0909/react-agent.git
cd react-agent
cp .env.example .env       # 填入 DEEPSEEK_API_KEY / QWEATHER_API_KEY
docker compose up -d
```

启动后访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

### 方式二：本地

```bash
pip install -r requirements.txt
cp .env.example .env       # 填入 key
# 把 embedding + reranker 模型放到项目根对应目录
python run_api.py
```

---

## 使用

### 流式聊天

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "电压是什么？", "thread_id": "user-001"}'
```

SSE 事件类型：

| event | 说明 |
|---|---|
| `tool_call` | LLM 决定调用工具 |
| `tool_result` | 工具返回结果 |
| `token` | 回答片段 |
| `done` | 本轮结束 |
| `error` | 异常 |

### 运行评测

```bash
python scripts/run_eval.py
```

报告输出到 `reports/eval_*.json`，包含 4 项指标 + 每条用例详情。

### 单元测试

```bash
pytest tests/ -v
```

---

## 项目结构

```
react-agent/
├── app/
│   ├── api/                    # FastAPI 路由 + SSE
│   │   ├── app.py              # FastAPI 实例 + lifespan（含 reranker 预热）
│   │   ├── deps.py             # 依赖注入
│   │   ├── schemas.py          # 请求/响应模型
│   │   └── routes/
│   │       ├── chat.py         # SSE 流式聊天
│   │       └── health.py       # 健康检查
│   ├── graph/                  # LangGraph 编排
│   │   ├── state.py            # AgentState
│   │   ├── nodes.py            # agent_node（LLM 推理）
│   │   ├── builder.py          # 图构建 + RetryPolicy
│   │   └── checkpointer.py     # AsyncSqliteSaver 生命周期
│   ├── rag/                    # RAG 流水线
│   │   ├── embedding.py        # 本地 HuggingFace Embedding
│   │   ├── vectorstore.py      # Chroma 持久化
│   │   ├── retriever.py        # BM25 + 向量混合检索
│   │   ├── query_rewrite.py    # LLM 查询改写
│   │   └── rerank.py           # Qwen3-Reranker 精排
│   ├── tools/                  # 工具集
│   │   ├── calculator.py       # AST 白名单安全计算器
│   │   ├── weather.py          # 和风天气 API（异步）
│   │   └── knowledge.py        # 知识库检索（RAG 入口）
│   ├── config.py               # Pydantic Settings 集中配置
│   ├── llm.py                  # LLM 单例
│   └── cli.py                  # CLI 调试入口
├── eval/                       # 评测
│   ├── dataset.py              # JSONL 数据集加载
│   ├── metrics.py              # 4 项指标
│   ├── runner.py               # 评测执行器
│   └── datasets/golden.jsonl   # 金标准 9 条
├── scripts/run_eval.py         # 评测入口
├── tests/                      # 单元测试（13 个）
├── docs/demo.txt               # 知识库文档
├── reports/                    # 评测报告输出
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── run_api.py
```

---

## 踩坑记录

做这个项目时踩过并解决的 4 个典型问题，也是面试常考的：

### 1. 异步 checkpointer 不兼容

**现象**：`SqliteSaver` 报 `NotImplementedError: The SqliteSaver does not support async methods`
**原因**：同步和异步 checkpointer 是两套类。`astream` 走异步图，必须用 `AsyncSqliteSaver`。
**修复**：`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.from_conn_string()` + `aiosqlite` 依赖。

### 2. 工具异常污染对话历史

**现象**：工具执行失败后，下次请求 DeepSeek 报 400 `An assistant message with 'tool_calls' must be followed by tool messages`
**原因**：工具抛异常 → 图被中断 → AIMessage（含 tool_calls）已入库，但对应 ToolMessage 未写入 → 历史不完整。
**修复**：`ToolNode(handle_tool_errors=...)` 兜底所有异常，转成字符串作为 ToolMessage 返回，**绝不 raise**。

### 3. Qwen3-Reranker 阈值不适用

**现象**：知识库拒答（`rag_miss` 用例）失败，无关问题也返回内容。
**原因**：Qwen3-Reranker 的 logits 分布与 bge-reranker 完全不同，固定阈值 `-6.0` 失效。
**修复**：改用相对分数（`top_score - margin`）+ LLM 相关性判断（yes/no）双重过滤。

### 4. 思考文本泄漏到最终答案

**现象**：答案开头出现 "I'll search the knowledge base..." 和查询改写变体。
**原因**：流式模式下，LLM 调工具前输出的「思考」content 在 tool_calls 拼装完成之前就被推送给用户。
**修复**：按轮缓冲，等 `updates` 事件确认这一轮是否产出 `tool_calls` 后再决定是否推送缓冲内容。

### 5. GPU 环境构建

**现象**：多次构建失败，报 `apt-get exit 100`、`torch==2.6.0 找不到`、`transformers 不识别 qwen3`、`init_empty_weights not defined`。

**修复路径**：
1. `nvidia/cuda` 基础镜像的 apt 源替换为清华镜像，移除过期的 NVIDIA 仓库列表
2. PyTorch 用 2.5.1+cu121（国内镜像源尚未同步 2.6.0），`transformers` 升到 4.51.0（识别 Qwen3 的最低要求）
3. 用约束文件 `pins.txt` 锁定 torch + transformers 版本，避免多依赖冲突
4. 补装 `accelerate` 解决 `init_empty_weights not defined` 报错
5. `pip install` 加大 `--timeout 1200 --retries 10 --resume-retries 10` 防止 780MB 的 torch 包下载中断

---

## Roadmap

- [ ] Langfuse 可观测（trace / token 成本 / prompt 版本对比）
- [ ] 多轮对话摘要（长对话压缩）
- [ ] 混合检索权重可调 + 自动化调参
- [ ] 评测集扩充到 50+ 条
- [ ] GitHub Actions CI

---

## License

MIT