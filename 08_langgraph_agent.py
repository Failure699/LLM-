# 08_langgraph_agent.py —— 把 07_agent.py 的 while True 循环，画成一张 LangGraph 图
import os
import chromadb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除，不会进仓库）
# ── ① embedding：硅基流动（照旧，跟 07 一样）──
embed_client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                    api_key=os.getenv("SILICONFLOW_API_KEY"))

def embed(text):
    return embed_client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})

# ── ② 工具：还是那两个函数，但 docstring 会被框架自动读成工具说明 ──
def search_notes(question: str) -> str:
    """在用户的私人笔记里检索相关段落。用户问题涉及笔记内容时先调用它查资料。"""
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    if results["distances"][0][0] > 0.9:
        return "资料中没有相关内容"
    return "\n\n".join(results["documents"][0])

def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    return f"{city}, 40°C, 晴"

tools = [search_notes, get_weather]

# ── ③ 模型：DeepSeek，bind_tools 把工具绑上去（LangChain 会自动生成工具 JSON）──
model = ChatOpenAI(base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model="deepseek-chat")
model_with_tools = model.bind_tools(tools)

# ── ④ 节点：每个节点 = 一个函数，收 state、还 state ──
def call_model(state):
    messages = [SystemMessage("你是问答助手，首先根据笔记里的内容回答")] + state["messages"]
    return {"messages": [model_with_tools.invoke(messages)]}
# ── ⑤ 建图：先立架子，再把边连起来 ──
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))   # ToolNode = 框架现成的"工具站"
builder.add_edge(START, "model")             # 起点 → 模型站（这条我帮你写好了）

# TODO① 你来写：路由函数 should_continue —— 对应 07 里的 if msg.tool_calls
# 它接收 state，读最后一条消息，判断模型有没有要调工具：
#   有 tool_calls → return "tools"（去工具站）
#   没有         → return END   （去终点）
def should_continue(state):
    last = state["messages"][- 1]
    if last.tool_calls:
        return "tools"
    return END

# TODO② 你来写：把剩下的两条边补上（补在这两个 TODO 下面、app = builder.compile() 之前）
# 1) 模型站出来要"分岔"：用 add_conditional_edges，靠 should_continue 决定去哪
# 2) 工具站跑完要"回模型站"：用 add_edge，这就是那个 while 环
builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = builder.compile(checkpointer=checkpointer)


# ── ⑥ 跑起来 ──
config = {"configurable": {"thread_id": "1"}}   # 对话编号，同一个 id = 同一段记忆
while True:
    ques = input("你的问题：")
    if ques.strip() in ("quit", "exit", "q", "退出"):
        break
    result = app.invoke({"messages": [("user", ques)]}, config=config)
    print("模型回答：", result["messages"][-1].content) # 一句话打完，补个换行
"""
result = app.invoke({"messages": [("user", "rag是啥")],
                    }, config=config)
print("① state 的类型：", type(result).__name__)        # 是不是 dict？
print("② state 有几个格子：", list(result.keys()))       # 是不是只有 messages？
print("③ messages 有几条：", len(result["messages"]))   # 是不是 5 条？

for i, m in enumerate(result["messages"]):
    print(f"[{i}] {type(m).__name__:14s} -> {repr(m.content)[:50]}")"""