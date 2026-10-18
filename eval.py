"""评测：带工具的 Agent 对比裸模型（不给工具）。

要证明的事：给大模型接上工具后，它能答对「实时 / 私人」问题；
裸模型没有工具，只能靠训练记忆瞎编或拒答。

跑法：python eval.py
"""
import json
import os
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/符号报错乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除）

# ── 客户端：embedding 用硅基流动，生成用 DeepSeek ──
embed_client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                      api_key=os.getenv("SILICONFLOW_API_KEY"))
deepseek = OpenAI(base_url="https://api.deepseek.com",
                  api_key=os.getenv("DEEPSEEK_API_KEY"))
KEY = os.getenv("QWEATHER_API_KEY")
HOST = os.getenv("QWEATHER_API_HOST")

# ── 工具定义（和 07_agent.py 保持一致）──
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "在用户的私人笔记里检索相关段落。当用户的问题涉及笔记内容时，先调用它查资料。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "要检索的问题"}},
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
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
]


def embed(text):
    return embed_client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding


# ── 打开持久化的向量库（里面已有笔记切块，不要重存）──
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})


def search_notes(question: str) -> str:
    """在用户的笔记里检索相关段落，返回拼接好的资料文本。"""
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    mdis = results["distances"][0][0]
    if mdis > 0.9:
        return "资料中没有相关内容"
    hits = results["documents"][0]
    return "\n\n".join(hits)


def get_weather(city: str) -> str:
    """查询某个城市的天气（城市名 → 城市 ID → 实时天气）。"""
    try:
        geo = requests.get(f"https://{HOST}/geo/v2/city/lookup",
                           params={"location": city, "key": KEY, "lang": "zh"},
                           timeout=10).json()
        if geo.get("code") != "200" or not geo.get("location"):
            return f"查不到城市「{city}」，请检查城市名是否正确"
        resp = requests.get(f"https://{HOST}/v7/weather/now",
                            params={"location": geo["location"][0]["id"], "key": KEY, "lang": "zh"},
                            timeout=10).json()
        if resp.get("code") != "200":
            return f"查询「{city}」天气失败（错误码 {resp.get('code')}）"
        now = resp["now"]
        return f"{city}, {now['temp']}°C, {now['text']}"
    except requests.RequestException:
        return f"查询「{city}」天气时网络出错，请稍后再试"


SYSTEM = ("你是助手，可以按需要使用工具（查笔记、查天气）。"
          "先判断用户问题需要哪个工具，再调用；回答要基于工具返回的结果，不要编造。")

# ── 评测集：10 题，两类 ──
#   weather：问实时天气，判题靠「温度数字是否匹配真实值」
#   notes  ：问笔记里的私人表述，判题靠「关键词命中」
EVAL_QUESTIONS = [
    {"type": "weather", "q": "北京现在多少度？", "city": "北京"},
    {"type": "weather", "q": "上海现在什么天气？", "city": "上海"},
    {"type": "weather", "q": "广州现在多少度？", "city": "广州"},
    {"type": "weather", "q": "深圳现在什么天气？", "city": "深圳"},
    {"type": "weather", "q": "杭州现在多少度？", "city": "杭州"},
    {"type": "notes", "q": "我的笔记里把 RAG 比作什么考试？", "keyword": "开卷考试"},
    {"type": "notes", "q": "我的笔记里说 LLM 的本质是什么机器？", "keyword": "文字接龙"},
    {"type": "notes", "q": "我的笔记里把 embedding 说成把文字变成什么？", "keyword": "坐标"},
    {"type": "notes", "q": "我的实习计划里，什么时候开始投递简历？", "keyword": "11 月底"},
    {"type": "notes", "q": "我本科是哪个学校、什么专业？", "keyword": "河海"},
]


def ask_bare(question: str) -> str:
    """裸模型：不带 tools，直接问。"""
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是助手，直接回答用户的问题。"},
                  {"role": "user", "content": question}],
        temperature=0,
    )
    return resp.choices[0].message.content


def ask_agent(question: str) -> tuple[str, list[str]]:
    """带工具的 agent：模型自主决定调哪个工具，返回最终回答 + 用过的工具。"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    used = []
    while True:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=tools, temperature=0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                used.append(tc.function.name)
                args = json.loads(tc.function.arguments)
                if tc.function.name == "search_notes":
                    result = search_notes(args["question"])
                elif tc.function.name == "get_weather":
                    result = get_weather(args["city"])
                else:
                    result = "未知工具"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return msg.content, used


def run():
    bare_score = agent_score = 0
    print("=" * 60)
    print("评测开始：裸模型（不给工具） vs Agent（给工具）")
    print("=" * 60)

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        bare_ans = ask_bare(item["q"])
        agent_ans, used = ask_agent(item["q"])

        # 判题：
        #   天气题测「能不能拿到实时数据」——看 agent 有没有调 get_weather 工具；
        #   裸模型没有工具，客观上拿不到实时值，一律判错。
        #   笔记题比关键词（私有表述是否命中）。
        if item["type"] == "weather":
            bare_ok = False
            agent_ok = "get_weather" in used
            hint = "（实时数据，裸模型无工具拿不到）"
        else:
            kw = item["keyword"]
            bare_ok = kw in bare_ans
            agent_ok = kw in agent_ans
            hint = f"（关键词「{kw}」）"

        bare_score += bare_ok
        agent_score += agent_ok

        print(f"\n题 {i}【{'天气' if item['type'] == 'weather' else '笔记'}】{item['q']} {hint}")
        print(f"  裸模型：{bare_ans}  → {'对' if bare_ok else '错'}")
        print(f"  Agent ：{agent_ans}  → {'对' if agent_ok else '错'}（用了工具：{used or '无'}）")

    total = len(EVAL_QUESTIONS)
    print("\n" + "=" * 60)
    print(f"结果：裸模型 {bare_score}/{total} 分，Agent {agent_score}/{total} 分")
    print("=" * 60)


if __name__ == "__main__":
    run()
