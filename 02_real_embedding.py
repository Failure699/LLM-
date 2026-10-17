# 02_real_embedding.py —— 第 2 课：真·embedding（bge-m3，1024 维）
# 目标：把第 1 课的玩具 3 维向量，换成真实中文 embedding，
#       看真实向量长啥样，再用同一套余弦相似度算一次——公式一个字没变。

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ⚠️ 密钥现在放 .env 里（已被 .gitignore 排除），别提交到 git 仓库
API_KEY = os.getenv("SILICONFLOW_API_KEY")

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=API_KEY,
)

# 把三个词交给 bge-m3，让它吐向量
resp = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["猫", "狗", "汽车"],
)

# resp.data 和 input 一一对应
vecs = [d.embedding for d in resp.data]
cat, dog, car = vecs

print("向量维度：", len(cat))
print("「猫」向量的前 10 个数：", [round(x, 4) for x in cat[:10]])

# 复用第 1 课写过的余弦相似度（一模一样）
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))

print("猫 vs 狗   :", round(cosine(cat, dog), 4))
print("猫 vs 汽车 :", round(cosine(cat, car), 4))
