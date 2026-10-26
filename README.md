# 个人知识库 Agent

一个能「查笔记、查天气」的 AI Agent：给大模型接上自己的知识库和工具，让它自主判断该用哪个工具来回答用户的问题。

## 能做什么
- 查笔记：用 RAG（检索增强生成）从私人笔记里检索相关内容，再据此回答
- 查天气：调用真实天气 API（和风天气，城市名 → 城市 ID → 实时天气）

## 评测结果（为什么它比裸模型强）

给大模型接上工具后，它能答对「实时 / 私人」问题，而**裸模型（不给工具）一道都答不上来**。

用 10 道题同时考「裸模型」和「本 Agent」（`python eval.py`）：

| 类型 | 裸模型 | Agent |
|------|--------|-------|
| 实时天气（5 题） | 0/5（拒答，拿不到实时数据） | 5/5（调用天气工具） |
| 私人知识（5 题） | 0/5（不知道你的笔记/个人情况） | 5/5（检索知识库） |
| **合计** | **0/10** | **10/10** |

判据：天气题看「是否调用天气工具」（能否拿到实时数据）；笔记题看「私有表述是否命中」。
### 工具路由与参数评测

使用 `eval_tool_routing.py` 对 Agent 的第一轮工具选择进行独立评测，不实际执行工具，从而排除网络和外部 API 的干扰。

| 指标 | 测试规模 | 结果 |
|------|----------|------|
| 工具种类精确匹配 | 15 个问题 × 3 轮 | 45/45（100%） |
| 天气工具调用次数与城市参数 | 4 个问题 × 3 轮 | 12/12（100%） |

路由测试覆盖：

- `search_notes`、`get_weather`、`remember`、`recall`
- 无工具问题
- 多工具并行调用
- “天气 API 笔记”与“实时天气”的边界
- 多城市参数及混合工具场景

运行：
```powershell
python eval_tool_routing.py
```
这组指标和原来的 `裸模型 0/10 vs Agent 10/10` 含义不同：

- 原评测：最终任务能否完成。
- 新评测：第一轮工具选择、调用次数和参数是否正确。

## 技术栈
- 大模型：DeepSeek（`deepseek-chat`）
- Embedding：硅基流动（`BAAI/bge-m3`）
- 向量库：chromadb
- Agent 框架：function calling / LangGraph
- 天气 API：和风天气（QWeather）
- 工具服务：MCP（stdio）

## 操作流程
1. 装依赖：`pip install openai chromadb python-dotenv langgraph mcp`
2. 配密钥：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`、`QWEATHER_API_KEY` 和 `QWEATHER_API_HOST`
3. 启动：
   - `python 07_agent.py` —— 多工具 Agent（查笔记 + 查天气）
   - `python eval.py` —— 评测：裸模型 vs Agent（10 题对比）
   - `python chat.py` —— 最简多轮对话
   - `python 09_mcp_client.py` —— 动态连接多个 MCP Server 的交互式 Agent
   - `python eval_tool_routing.py` —— 工具路由与参数稳定性评测

## 目录结构
- `01` ~ `06`：RAG 学习路径（余弦相似度 → embedding → 切块 → 检索 → chroma 持久化）
- `07_agent.py`：多工具 function-calling agent
- `08_langgraph_agent.py`：LangGraph 状态机 agent
- `09_mcp_client.py`：动态加载 MCP Server，支持多工具并发、短期对话与用户记忆隔离
- `mcp_memory.py`：基于 ChromaDB 的长期记忆 MCP 服务
- `eval_tool_routing.py`：15题工具路由与4题天气参数评测，自动重复3轮
- `eval.py`：评测脚本（裸模型 vs Agent，10 题对比）
- `mcp_weather.py` / `mcp_notes.py`：把工具抽成 MCP 服务（对话窗口里 `/mcp` 连接）
- `hello_llm.py`：第一个脚本，调一次大模型、打印回答
- `chat.py`：多轮对话版机器人（带上下文记忆）
