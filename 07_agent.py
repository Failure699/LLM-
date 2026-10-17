import json
import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除，不会进仓库）

# ── ① 两个客户端：embedding 用硅基流动，生成用 DeepSeek ──
embed_client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                    api_key=os.getenv("SILICONFLOW_API_KEY"))
deepseek = OpenAI(base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY"))
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "在用户的私人笔记里检索相关段落。当用户的问题涉及笔记内容时，先调用它查资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要检索的问题"}
                },
                "required": ["question"],
            },
        },
    },
    {
      "type": "function",
      "function": {
          "name": "get_weather",
          "description": "查询某个城市的天气",
          "parameters": {
              "type": "object",
              "properties": {
                  "city": {"type": "string", "description": "城市名"}
              },
              "required": ["city"],
            },
        },
    },
]

def embed(text):
    return embed_client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding

# ── ② 打开昨天建好的库（里面已有 19 块，不要再重存）──
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})

# ── ③ TODO 你来写：把「查资料」打包成一个工具函数 ──
def search_notes(question: str) -> str:
    """在用户的笔记里检索相关段落，返回拼接好的资料文本。"""
    # 1) q_vec = embed(question)
    # 2) results = col.query(query_embeddings=[q_vec], n_results=3)
    # 3) 取第一名距离 distances[0][0]，若 > 0.9 就 return "资料中没有相关内容"
    # 4) 否则 return "\n\n".join(hits)   （hits 是命中的正文列表）
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    mdis = results["distances"][0][0]
    if mdis > 0.9:
        return "资料中没有相关内容"
    else:
        hits = results["documents"][0]
        return "\n\n".join(hits)
def get_weather(city: str) -> str:
    return f"{city}, 40°C, 晴"  # 这里可以接真实天气 API

messages = [
        {"role": "system", "content": "你是助手，可以按需要使用工具（查笔记、查天气）。先判断用户问题需要哪个工具，再调用；回答要基于工具返回的结果，不要编造。"},
    ]
def ask(question: str) -> str:
    messages.append({"role": "user", "content": question})
    while True:
        response = deepseek.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=tools,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                print("🔧 模型要调:", tc.function.name, "参数:", tc.function.arguments)
                args = json.loads(tc.function.arguments)
                if tc.function.name == "search_notes":
                    result = search_notes(args["question"])
                elif tc.function.name == "get_weather":
                    result = get_weather(args["city"])
                else:
                    result = "未知工具"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
           messages.append(msg)
           return msg.content          # ← TODO② 原来是 print + break，改成 return

while True:
    ques = input("你的问题")
    if ques.strip() in ("quit", "exit", "q", "退出"):   # ← 新加
        break
    ans = ask(ques)
    print("模型回答：", ans)
# ── TODO③ 你来写：交互式，让你能连续问 ──
# 提示：while True 循环 + input("你问：") 拿输入，然后 print(ask(输入))