# 01_cosine_similarity.py —— 第 1 课：向量相似度（零依赖，不用装任何包）
# 目标：把笔记里那句"猫狗近、汽车远"变成能跑的代码，
#       亲手写出「余弦相似度」这个 RAG 里最核心的计算。

import math

# 玩具 embedding：每个词用一个 3 维向量表示（真实模型是几百上千维，道理一样）
# 注意看数值：猫和狗的每一维都差不多大，汽车前两维很小、第三维很大
vectors = {
    "猫":   [1,0, 0],
    "狗":   [0.88, 0.78, 0.09],
    "汽车": [- 1, 0, 0],
    "自行车" : [0.20, 0.20, 0.15]
}


def dot(a, b):
    """点积：对应位相乘再相加。两向量越"同向"，点积越大。"""
    ans = 0
    for i in range(len(a)):
        ans += a[i] * b[i]
    return ans


def norm(a):
    """模长（向量长度）：sqrt(各维平方和)。只用来做分母，消除长度影响。"""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a, b):
    """余弦相似度 = 点积 / (模长 × 模长)
    范围 [-1, 1]：1 = 方向完全一样，0 = 垂直，-1 = 完全相反。
    RAG 检索时，就是拿问题向量和每个文档向量算这个，取最接近 1 的几篇。"""
    return dot(a, b) / (norm(a) * norm(b))


# 验证：猫 vs 狗 应该接近 1（都是宠物），猫 vs 汽车 应该低很多
print("猫 vs 狗   :", round(cosine_similarity(vectors["猫"], vectors["狗"]), 4))
print("猫 vs 汽车 :", round(cosine_similarity(vectors["猫"], vectors["汽车"]), 4))
print("猫 vs 自行车 :", round(cosine_similarity(vectors["猫"], vectors["自行车"]), 4))


# ============ 你来改（练完这 4 个才算会） ============
# 1. 加一个词"自行车"，给它一个 3 维向量，猜猜它跟谁更近，再验证。
# 2. 把 dot() 里的 zip 换成手写 for 循环，确认你真的懂点积在干嘛。
# 3. 把"猫"和"汽车"换成两个完全相反方向的向量（比如 [1,0,0] 和 [-1,0,0]），
#    看看余弦相似度会不会变成 -1。
# 4. 问自己：为什么除以模长？如果把模长删掉（直接返回点积），
#    一个很"长"的向量会不会假装自己跟谁都像？—— 这就是余弦相似度的意义。C:\Users\Dashachun\Project\llm-lear\01_cosine_similarity.py
