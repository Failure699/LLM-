import glob
import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY")   # 从 .env 读，别写死

client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=API_KEY)

def embed(text):
    return client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding

def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

# 1. 建库
chroma = chromadb.Client()
col = chroma.create_collection("my_notes")

# 2. 读 notes 文件夹所有 .md 文件，切块 + 存（这段我来，你要读懂每行）
files = glob.glob("notes/*.md")                 # ← 新：列出所有笔记文件路径
if col.count() == 0:  # 如果数据库为空，则进行存储
    for f in files:
        text = open(f, encoding="utf-8").read()     # ← 新：读整个文件内容
        chunks = chunk_text(text)                   # 切成块
        for i, c in enumerate(chunks):              # i 是第几块
            col.add(
                ids=[f"{f}#{i}"],                   # 每块唯一编号：文件名#第几块
                documents=[c],
                embeddings=[embed(c)],
                metadatas=[{"source": f}],          # ← 新：记来源（这段来自哪个文件）
            )

print("已存入", col.count(), "块")

# 3. 查询（TODO 你来写）
question = "怎么用两个鸡蛋一个香蕉制作一个斐济背？"
q_vec = embed(question)
results = col.query(query_embeddings=[q_vec], n_results=3)
hits = results["documents"][0]   # 命中的正文列表
metas = results["metadatas"][0]  # 命中的来源，形如 [{'source': 'notes/note2_rag.md'}, ...]
for hit, meta in zip(hits, metas):
    print(f"[{meta['source']}]\n {hit}\n" )
# 提示：
#   hits  = results["documents"][0]   命中的正文列表
#   metas = results["metadatas"][0]   命中的来源，形如 [{'source': 'notes/note2_rag.md'}, ...]
#   用 zip(hits, metas) 逐条打印，来源取 meta['source']
deepseek = OpenAI(
      base_url="https://api.deepseek.com",
      api_key=os.getenv("DEEPSEEK_API_KEY"),
  )
context = "\n\n".join(hits)
messages = [{"role": "system", "content": "你是问答助手，只能根据下面提供的【资料】回答。如果资料里没有相关信息，只回答'资料中没有相关内容'，不要解释、不要猜测、不要编造。"},

            {"role": "user", "content": f"上下文信息：{context}\n\n问题：{question}"}]
response = deepseek.chat.completions.create(
    model = "deepseek-chat",
    messages = messages,
)
print("模型回答：", response.choices[0].message.content)
