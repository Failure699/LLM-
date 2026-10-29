# 个人知识库 MCP Agent

一个面向个人学习场景的多工具 Agent。它能够判断用户意图，按需查询私人笔记、实时天气和长期记忆，再基于真实工具结果生成回答；同时使用 critic 节点检查答案，失败时自动删除草稿并重试。

这个项目主要解决三个问题：

- 大模型不知道私人资料：用 RAG 检索本地学习笔记。
- 大模型拿不到实时信息：通过 MCP 接入天气 API。
- 普通聊天重启后会失忆：组合 SQLite 短期记忆与 ChromaDB 长期记忆。

## 核心能力

- **动态工具发现**：从 `.mcp.json` 启动多个 MCP Server，自动获取工具说明与参数结构。
- **多工具并发**：一个问题需要多个工具时，通过 `asyncio.gather()` 并发执行。
- **私人笔记 RAG**：使用 `BAAI/bge-m3` 向量化笔记，ChromaDB 持久化存储并返回来源引用。
- **双层记忆**：LangGraph checkpointer 保存线程内对话；`remember` / `recall` 保存跨会话长期记忆。
- **用户隔离**：`user_id` 不交给模型生成，由客户端注入可信值，避免读取其他用户的记忆。
- **回答反思**：critic 根据对话历史和工具证据判断回答是否正确、有用；失败草稿会被删除并重新生成。
- **循环保护**：限制 critic 重试次数和 LangGraph 总执行步数，防止无限循环。

## 系统架构

```text
用户问题
   ↓
LangGraph model 节点（DeepSeek）
   ├─ 不需要工具 ─────────────────────┐
   │                                  ↓
   └─ 生成 tool_calls → MCP tools → model
                         │             ↓
                         ├─ search_notes（私人笔记 RAG）
                         ├─ get_weather（实时天气）
                         ├─ remember（写入长期记忆）
                         └─ recall（检索长期记忆）
                                       ↓
                                critic 评审节点
                                  ├─ 通过 → 最终回答
                                  └─ 失败 → 删除草稿并重试
```

MCP Client 读取 `.mcp.json`，通过 stdio 启动并连接三个 Server：

| MCP Server | 工具 | 作用 |
|---|---|---|
| `mcp_weather.py` | `get_weather` | 查询实时天气 |
| `mcp_notes.py` | `search_notes` | 检索私人学习笔记并返回来源 |
| `mcp_memory.py` | `remember`、`recall` | 保存和检索跨会话记忆 |

## 评测结果

### 最终任务效果

使用 10 道实时信息与私人知识问题对比裸模型和 Agent（`python eval.py`）：

| 类型 | 裸模型 | Agent |
|---|---:|---:|
| 实时天气（5 题） | 0/5 | 5/5 |
| 私人知识（5 题） | 0/5 | 5/5 |
| **合计** | **0/10** | **10/10** |

判据：天气题检查是否获得实时工具数据；笔记题检查是否命中私人知识库中的目标内容。

### 工具路由与参数

`eval_tool_routing.py` 只评测第一轮工具选择，不实际执行外部工具，从而排除网络波动：

| 指标 | 测试规模 | 结果 |
|---|---:|---:|
| 工具种类精确匹配 | 15 个问题 × 3 轮 | 45/45（100%） |
| 天气调用次数与城市参数 | 4 个问题 × 3 轮 | 12/12（100%） |

测试覆盖 `search_notes`、`get_weather`、`remember`、`recall`、无工具问题、多工具组合、多城市参数，以及“天气 API 笔记”和“实时天气”的意图区分。

> 以上结果来自当前自建小规模测试集，用于回归验证，不代表生产环境中的泛化准确率。

## 技术栈

- 生成模型：DeepSeek `deepseek-chat`
- Embedding：硅基流动 `BAAI/bge-m3`
- 向量数据库：ChromaDB PersistentClient
- Agent 编排：LangGraph
- 工具协议：MCP（stdio）
- 短期记忆：LangGraph `AsyncSqliteSaver`
- 天气数据：和风天气 API

## 快速开始

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install openai langchain-openai langgraph langgraph-checkpoint-sqlite chromadb mcp python-dotenv requests
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填写：

```dotenv
DEEPSEEK_API_KEY=你的密钥
SILICONFLOW_API_KEY=你的密钥
QWEATHER_API_KEY=你的密钥
QWEATHER_API_HOST=你的API_HOST
NOTES_DIR=notes
```

`.env`、ChromaDB 数据和 SQLite 对话记录均已加入 `.gitignore`，不会提交到仓库。

### 3. 准备私人笔记索引

将 Markdown 笔记放入 `NOTES_DIR` 指定的目录，然后运行：

```powershell
python 06_rag_persistent.py
```

首次运行或笔记发生变化时更新索引；日常查询可以直接复用已有 ChromaDB。

### 4. 启动完整 Agent

```powershell
python 10_langgraph_mcp_agent.py
```

程序会根据 `.mcp.json` 自动启动 MCP Server。输入 `q`、`quit`、`exit` 或 `退出` 结束。

运行前可在 `10_langgraph_mcp_agent.py` 中修改：

- `USER_ID`：长期记忆所属用户。
- `thread_id`：当前短期对话线程；相同值恢复历史，不同值创建隔离会话。

### 5. 运行评测

```powershell
python eval.py
python eval_tool_routing.py
python 06_rag_threshold_eval.py
```

## 主要文件

| 文件 | 作用 |
|---|---|
| `01`～`06` | 从余弦相似度、Embedding、分块到持久化 RAG 的学习路径 |
| `06_rag_threshold_eval.py` | 用正负样本评估 RAG 距离阈值 |
| `07_agent.py` | 原生 function calling 多工具 Agent |
| `08_langgraph_agent.py` | LangGraph 状态、条件路由与 critic |
| `09_mcp_client.py` | 原生异步 MCP Client、多 Server 和并发工具调用 |
| `10_langgraph_mcp_agent.py` | 最终版：LangGraph + MCP + 双层记忆 + critic |
| `mcp_weather.py` | 天气 MCP Server |
| `mcp_notes.py` | 私人笔记 RAG MCP Server |
| `mcp_memory.py` | 长期记忆 MCP Server |
| `eval_tool_routing.py` | 工具种类、调用次数和参数评测 |
| `eval.py` | 裸模型与 Agent 的最终任务对比 |

## 当前边界

- RAG 阈值来自当前笔记和小规模测试集，换语料后需要重新评估。
- stdio MCP Server 适合本地单机演示，尚未实现远程部署、鉴权和服务监控。
- 当前用户与线程标识由本地常量配置；生产系统应由登录态或服务端身份系统生成。
- 天气结果依赖外部 API 的可用性和数据范围。
