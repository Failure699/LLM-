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

## 技术栈
- 大模型：DeepSeek（`deepseek-chat`）
- Embedding：硅基流动（`BAAI/bge-m3`）
- 向量库：chromadb
- Agent 框架：function calling / LangGraph
- 天气 API：和风天气（QWeather）
- 工具服务：MCP（stdio）

## 怎么跑起来
1. 装依赖：`pip install openai chromadb python-dotenv langgraph mcp`
2. 配密钥：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`、`QWEATHER_API_KEY` 和 `QWEATHER_API_HOST`
3. 启动：
   - `python 07_agent.py` —— 多工具 Agent（查笔记 + 查天气）
   - `python eval.py` —— 评测：裸模型 vs Agent（10 题对比）
   - `python chat.py` —— 最简多轮对话

## 目录结构
- `01` ~ `06`：RAG 学习路径（余弦相似度 → embedding → 切块 → 检索 → chroma 持久化）
- `07_agent.py`：多工具 function-calling agent
- `08_langgraph_agent.py`：LangGraph 状态机 agent
- `eval.py`：评测脚本（裸模型 vs Agent，10 题对比）
- `mcp_weather.py` / `mcp_notes.py`：把工具抽成 MCP 服务（对话窗口里 `/mcp` 连接）
- `hello_llm.py`：第一个脚本，调一次大模型、打印回答
- `chat.py`：多轮对话版机器人（带上下文记忆）
