import json
import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import requests
load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除，不会进仓库）
KEY = os.getenv("QWEATHER_API_KEY")
HOST = os.getenv("QWEATHER_API_HOST")
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

def search_notes(question: str) -> str:
    """在用户的笔记里检索相关段落，返回拼接好的资料文本。"""
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    mdis = results["distances"][0][0]
    if mdis > 0.9:
        return "资料中没有相关内容"
    else:
        hits = results["documents"][0]
        return "\n\n".join(hits)
def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    try:
        # 第 1 步：城市名 → 城市 ID（lang=zh 让天气描述统一返回中文，不随当地语言变）
        geo = requests.get(f"https://{HOST}/geo/v2/city/lookup",
                            params={"location": city, "key": KEY, "lang": "zh"},
                            timeout=10).json()
        # 查不到城市时 code 不是 200，location 也是空列表，给友好提示而不是崩
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
        # 断网 / 超时 / 域名不通等网络层错误，别让工具直接抛异常
        return f"查询「{city}」天气时网络出错，请稍后再试"

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
           return msg.content
if __name__ == "__main__":
    while True:
        ques = input("你的问题")
        if ques.strip() in ("quit", "exit", "q", "退出"):
            break
        ans = ask(ques)
        print("模型回答：", ans)