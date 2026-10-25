import glob
import os
import sys
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY")   # 从 .env 读，别写死

client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=API_KEY)

def embed_batch(texts):
    response = client.embeddings.create(
        model = "BAAI/bge-m3",
        input = texts
    )

    return [
        item.embedding
        for item in response.data
    ]

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

# 1. 建库
chroma = chromadb.PersistentClient(path="./chroma_db")
#chroma.delete_collection("my_notes")
col = chroma.get_or_create_collection("my_notes", metadata={"hnsw:space": "cosine"})
UPDATE_INDEX = False
DISTANCE_THRESHOLD = 0.5
if UPDATE_INDEX or col.count() == 0:
# 2. 读 notes 文件夹所有 .md 文件，切块 + 存（这段我来，你要读懂每行）
    NOTES_DIR = os.getenv("NOTES_DIR", "notes")

    files = glob.glob(
        os.path.join(NOTES_DIR, "*.md")
    )

    files = [
        file_path
        for file_path in files
        if os.path.basename(file_path) != "AGENTS.md"
    ]              # ← 新：列出所有笔记文件路径
    print("准备索引的文件数量：", len(files))
    for file_path in files:
        print("准备索引：", os.path.basename(file_path))
    for file_path in files:
        with open(file_path, encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_text(text)
        embeddings = embed_batch(chunks)

        source_name = os.path.basename(file_path)

        ids = [
            f"{source_name}#{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {"source": file_path}
            for _ in chunks
        ]

        col.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(
            f"已更新：{source_name}，"
            f"{len(chunks)} 个分块")
else:
    print("跳过索引更新，直接使用现有数据库")

print("数据库当前共有：", col.count(), "块")
# 3. 检索相关分块，并让 DeepSeek 根据资料回答
questions = "长期记忆最多可以保存多少GB"
q_vec = embed_batch( [questions])[0]
results = col.query(query_embeddings = [q_vec], n_results=3)
hits = results["documents"][0]   # 命中的正文列表
metas = results["metadatas"][0]  # 命中的来源，形如 [{'source': 'notes/note2_rag.md'}, ...]
#for hit, meta in zip(hits, metas):
    #print(f"[{meta['source']}]\n {hit}\n" )
# 提示：
distances = results["distances"][0]
relevant_hits = []
relevant_sources = set()
for hit, meta, distance in zip(
    hits,
    metas,
    distances
):
    print("\n距离：", distance)
    print("来源：", meta["source"])
    #print("内容：", hit)
    if distance <= DISTANCE_THRESHOLD:
        source_name = os.path.basename(meta["source"])

        evidence =f"来源： {source_name}\n{hit}"
        relevant_hits.append(evidence)
        relevant_sources.add(source_name)

print("通过阈值的分块数量：", len(relevant_hits))
if not relevant_hits:
    print("资料中没有相关内容")
else:
    context = "\n\n".join(relevant_hits)
#   hits  = results["documents"][0]   命中的正文列表
#   metas = results["metadatas"][0]   命中的来源，形如 [{'source': 'notes/note2_rag.md'}, ...]
#   用 zip(hits, metas) 逐条打印，来源取 meta['source']
    deepseek = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    messages = [{"role": "system", "content": """你是问答助手，只能根据下面提供的【资料】回答。如果资料里没有相关信息，只回答'资料中没有相关内容'，不要解释、不要猜测、不要编造。
                """},
                {"role": "user", "content": f"上下文信息：{context}\n\n问题：{questions}"}]
    response = deepseek.chat.completions.create(
        model = "deepseek-chat",
        messages = messages,
    )
    answer = response.choices[0].message.content
    print("模型回答：", answer)
    if not answer.strip().startswith("资料中没有相关内容"):
        print("检索来源：")
        for source_name in sorted(relevant_sources):
            print("-", source_name)
