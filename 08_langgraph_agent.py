# 08_langgraph_agent.py —— 把 07_agent.py 的 while True 循环，画成一张 LangGraph 图
import os
import requests
import chromadb
import json
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

MAX_try = 3
load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除，不会进仓库）
KEY = os.getenv("QWEATHER_API_KEY")   # 和风天气
HOST = os.getenv("QWEATHER_API_HOST")
class agentstate(MessagesState):
    retry_count : int
    critique : str
    review_passed: bool
# ── ① embedding：硅基流动（照旧，跟 07 一样）──
embed_client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                    api_key=os.getenv("SILICONFLOW_API_KEY"))

def embed(text):
    return embed_client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})
col_memory = chroma.get_or_create_collection("my_memory", metadata={"hnsw:space": "cosine"})
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
    try:
        # 第 1 步：城市名 → 城市 ID（lang=zh 让天气描述统一返回中文）
        geo = requests.get(f"https://{HOST}/geo/v2/city/lookup",
                           params={"location": city, "key": KEY, "lang": "zh"},
                           timeout=10).json()
        if geo.get("code") != "200" or not geo.get("location"):
            return f"查不到城市「{city}」，请检查城市名是否正确"

        # 第 2 步：城市 ID → 实时天气
        resp = requests.get(f"https://{HOST}/v7/weather/now",
                            params={"location": geo["location"][0]["id"], "key": KEY, "lang": "zh"},
                            timeout=10).json()
        if resp.get("code") != "200":
            return f"查询「{city}」天气失败（错误码 {resp.get('code')}）"
        now = resp["now"]
        return f"{city}, {now['temp']}°C, {now['text']}"
    except requests.RequestException:
        return f"查询「{city}」天气时网络出错，请稍后再试"
def remember(text: str) -> str:
    """存储对话记忆"""
    vec = embed(text)
    col_memory.add(ids = [str(time.time())], documents= [text], embeddings= [vec])
    return f"存入成功: {text}"
def recall(question: str) -> str:
    """获取对话记忆"""
    q_vec = embed(question)
    results = col_memory.query(query_embeddings= [q_vec], n_results= 2)
    # ② 空库 → 也没有相关记忆
    if not results["documents"][0]:
        return "没有相关记忆"
    if results["distances"][0][0] > 0.9:
        return "没有相关记忆"
    return "\n\n".join(results["documents"][0])

tools = [search_notes, get_weather, remember, recall]

# ── ③ 模型：DeepSeek，bind_tools 把工具绑上去（LangChain 会自动生成工具 JSON）──
model = ChatOpenAI(base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model="deepseek-chat")
model_with_tools = model.bind_tools(tools)
critic_model = model.bind(
    response_format = {"type" : "json_object"}
)
def critic(state):
    old_count = state.get("retry_count", 0)
    answer_message = state["messages"][-1]
    answer = answer_message.content

    tool_results = []

    for m in reversed(state["messages"]):
        if m.type == "human":
            question = m.content
            break

        if m.type == "tool":
            tool_results.append(m.content)

    tool_results.reverse()
    if tool_results:
        tool_evidence = "\n".join(tool_results)
    else:
        tool_evidence = "本轮没有调用工具"
    message = [
        SystemMessage("""你是严格评审，逐条检查回答，两条都要看：
        ① 正确：回答内容本身对不对，有没有瞎编。
        ② 有用：回答对用户有没有实际帮助。如果用户的问题本身就没意义
        （比如只发一个数字、一个表情、无意义字符），或者回答只是敷衍、
        复读、绕圈子，就属于「没用」。
         只要有一条不满足，就判「错误」。
        必须只输出合法 JSON，不要输出 Markdown 或其他文字。
        JSON 格式示例：
        {
        "passed": true,
        "reason": "回答与工具证据一致"
        }。
        如果提供了工具证据，以工具证据为事实依据；
        回答与工具证据一致时，不要以“缺少数据来源”为由判错。"""),
        HumanMessage(f"问题：{question}\n 工具证明：{tool_evidence} \n回答：{answer}")
    ]
    review = critic_model.invoke(message)
    try:
        review_data = json.loads(review.content)
        passed = review_data["passed"]
        reason = review_data["reason"]

        if not isinstance(passed, bool):
            raise TypeError("passed必须是布尔值")
    except (json.JSONDecodeError, KeyError, TypeError):
        passed = False
        reason = "critic返回的JSON格式异常"
    if not passed:
        new_count = old_count + 1
        message_update = [
            RemoveMessage(id = answer_message.id)
        ]
    else:
        new_count = 0
        message_update = []
    #print(review.content)
    print(
    "评审：", reason,
    "是否通过：", passed,
    "retry_count：", new_count
    )
    return {"critique" : reason,
            "retry_count" : new_count,
            "messages" : message_update,
            "review_passed" : passed
            }
# ── ④ 节点：每个节点 = 一个函数，收 state、还 state ──
def call_model(state):

    passed =  state.get("review_passed", False)
    critique = state.get("critique", "")
    messages = [SystemMessage("""你是问答助手，首先根据笔记里的内容回答，
                                用户透露「关于自己」的重要信息（名字、学校、计划…）
                                → 调 remember 记下来回答需要「用户个人信息」时 → 先调 recall 查记忆，
                                再答回答要基于 recall 返回的结果，查不到就直说，别编""")] + state["messages"]
    if not passed and critique:
        messages.append(HumanMessage(f"上一版回答的评审意见：{critique}\n"
                "请根据评审意见重新回答用户原问题。"))
    return {"messages": [model_with_tools.invoke(messages)]}
# ── ⑤ 建图：先立架子，再把边连起来 ──
builder = StateGraph(agentstate)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))   # ToolNode = 框架现成的"工具站"
builder.add_edge(START, "model")             # 起点 → 模型站（这条我帮你写好了）
builder.add_node("critic", critic)

def should_continue(state):
    last = state["messages"][- 1]
    if last.tool_calls:
        return "tools"
    return "critic"
def after_critic(state):
    count = state.get("retry_count", 0)
    passed = state.get("review_passed", False)
    if passed or count >= MAX_try: # 评审说「正确」→ 交卷
        return END
    return "model"                            # 评审说「错误」→ 回去重写

builder.add_conditional_edges("model", should_continue, {"tools": "tools", "critic": "critic"})
builder.add_edge("tools", "model")
builder.add_conditional_edges("critic", after_critic, {"model" : "model", END : END})
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = builder.compile(checkpointer=checkpointer)


# ── ⑥ 跑起来 ──
config = {"configurable": {"thread_id": "8848"}}   # 对话编号，同一个 id = 同一段记忆
while True:
    ques = input("你的问题：")
    if ques.strip() in ("quit", "exit", "q", "退出"):
        break
    if not ques.strip():
        print("请输入具体问题")
        continue
    #result = app.invoke({"messages": [("user", ques)]}, config=config)
    #print("模型回答：", result["messages"][-2].content) # 一句话打完，补个换行
    result = app.invoke({"messages": [("user", ques)],
                         "retry_count" : 0,
                          "critique" : "",
                           "review_passed" : False },
                         config=config)
    passed = result["review_passed"]
    if not passed:
        print("模型回答： 太难了我不会")
    else:
        print("模型回答：", result["messages"][- 1].content)
