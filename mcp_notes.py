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
DISTANCE_THRESHOLD = 0.5
# ── ③ 开店 ──
server = MCPServer("notes")

# ── ④ 挂菜单：docstring 会变成工具的 description，签名只暴露 question ──
@server.tool()
def search_notes(question: str) -> str:
    """在用户的私人笔记里检索相关段落。用户问题涉及笔记内容时先调用它查资料。"""
    # 1) 将问题转成向量，并检索最相关的 3 个分块。
    # 2) 逐块使用距离门槛过滤，保留正文与来源的一一对应关系。
    # 3) 没有合格分块时明确拒答；否则返回正文和去重后的来源清单。
    q_vec = embed(question)
    results = col.query(query_embeddings=[q_vec], n_results=3)
    hits = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    relevant_hits = []
    relevant_sources = set()
    for hit, meta, distance in zip(hits, metas, distances):
        if distance <= DISTANCE_THRESHOLD:
            source_name = os.path.basename(meta["source"])
            evidence = f"来源为： {source_name}\n {hit}"
            relevant_hits.append(evidence)
            relevant_sources.add(source_name)
    if not relevant_hits:
        return "资料中没有相关内容"
    context = "\n\n".join(relevant_hits)
    source_text = "\n".join(f"- {source_name}"
            for source_name in sorted(relevant_sources))
    return f"资料内容：\n{context}\n\n检索来源：\n{source_text}"

# ── ⑤ 开门 ──
if __name__ == "__main__":
    server.run()
