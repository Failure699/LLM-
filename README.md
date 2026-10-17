# 个人知识库 Agent

一个能「查笔记、查天气」的 AI Agent：给大模型接上自己的知识库和工具，让它自主判断该用哪个工具来回答用户的问题。

## 能做什么
- 查笔记：用 RAG（检索增强生成）从私人笔记里检索相关内容，再据此回答
- 查天气：调用天气工具（当前是写死的演示数据「40°C 晴」，待接入真实 API）

## 技术栈
- 大模型：DeepSeek（`deepseek-chat`）
- Embedding：硅基流动（`BAAI/bge-m3`）
- 向量库：chromadb
- Agent 框架：function calling / LangGraph
- 工具服务：MCP（stdio）

## 怎么跑起来
1. 装依赖：`pip install openai chromadb python-dotenv langgraph mcp`
2. 配密钥：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY` 和 `SILICONFLOW_API_KEY`
3. 启动：
   - `python 07_agent.py` —— 多工具 Agent（查笔记 + 查天气）
   - `python chat.py` —— 最简多轮对话

## 目录结构
- `01` ~ `06`：RAG 学习路径（余弦相似度 → embedding → 切块 → 检索 → chroma 持久化）
- `07_agent.py`：多工具 function-calling agent
- `08_langgraph_agent.py`：LangGraph 状态机 agent
- `mcp_weather.py` / `mcp_notes.py`：把工具抽成 MCP 服务（对话窗口里 `/mcp` 连接）
- `hello_llm.py`：第一个脚本，调一次大模型、打印回答
- `chat.py`：多轮对话版机器人（带上下文记忆）
