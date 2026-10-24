
import os
import chromadb
import time
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
col_memory = chroma.get_or_create_collection("my_memory", metadata={"hnsw:space": "cosine"})

# ── ③ 开店 ──
server = MCPServer("memory")


@server.tool()
def remember(text: str, user_id: str) -> str:
    """存储对话记忆"""
    vec = embed(text)
    col_memory.add(ids = [str(time.time())], documents= [text],
                    embeddings= [vec], metadatas= [{"user_id": user_id}])
    return f"用户 {user_id} 存入成功：{text}"
@server.tool()
def recall(question: str, user_id: str) -> str:
    """获取对话记忆"""
    q_vec = embed(question)
    results = col_memory.query(query_embeddings= [q_vec], n_results= 2, where = {"user_id": user_id})
    # ② 空库 → 也没有相关记忆
    if not results["documents"][0]:
        return "没有相关记忆"
    if results["distances"][0][0] > 0.9:
        return "没有相关记忆"
    return results["documents"][0][0]

# ── ⑤ 开门 ──
if __name__ == "__main__":
    server.run()
