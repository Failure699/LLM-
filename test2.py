import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY")
client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=API_KEY)
def embed(text):
      #"""一句话 -> 1024 维向量"""
    return client.embeddings.create(model="BAAI/bge-m3", input=text).data[0].embedding
   
chroma = chromadb.Client()
col = chroma.create_collection("notes")
docs = ["猫是一种宠物", "狗是一种宠物", "汽车是交通工具", "火车是交通工具"]
ids = ["d1", "d2", "d3", "d4"]
vsc = [embed(doc) for doc in docs]
col.add(documents=docs,  ids=ids, embeddings=vsc)
query = "会哈气的动物"
query_embedding = embed(query)
results = col.query(query_embeddings=[query_embedding], n_results = 2)
print("query:", query)
hits  = results["documents"][0]   # ['狗是一种宠物', '猫是一种宠物']
dists = results["distances"][0]   # [0.8294, 0.9219]

print("最像的结果（越靠前越像）：")
for rank, (doc, dist) in enumerate(zip(hits, dists), start=1):
    print(f"  第{rank}名：{doc}　（距离 {dist:.3f}）")
