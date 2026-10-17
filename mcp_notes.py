# mcp_notes.py —— 把 search_notes 搬成 MCP 工具（知识库 Agent 的核心服务）
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer   # ① 导入（跟 mcp_weather.py 一样）

load_dotenv()

# ── ② 开店前先备货：embedding 客户端 + 向量库（search_notes 的外部依赖，放模块顶层）──
#   这些只会在 server 启动时跑一次，之后每次调用直接复用，不会重复联网/重复开库。
embed_client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                      api_key=os.getenv("SILICONFLOW_API_KEY"))

def embed(text):
    return embed_client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})

# ── ③ 开店 ──
server = MCPServer("notes")

# ── ④ 挂菜单：docstring 会变成工具的 description，签名只暴露 question ──
@server.tool()
def search_notes(question: str) -> str:
    """在用户的私人笔记里检索相关段落。用户问题涉及笔记内容时先调用它查资料。"""
    # TODO 你来写：把 07_agent.py 里 search_notes 的函数体抄过来（就 4 行逻辑）
    # 1) q_vec = embed(question)
    # 2) results = col.query(query_embeddings=[q_vec], n_results=3)
    # 3) 第一名距离 distances[0][0] > 0.9 → return "资料中没有相关内容"
    # 4) 否则 return "\n\n".join(results["documents"][0])
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    if results["distances"][0][0] > 0.9:
        return "资料中没有相关内容"
    return "\n\n".join(results["documents"][0])

# ── ⑤ 开门 ──
if __name__ == "__main__":
    server.run()
