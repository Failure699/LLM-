import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY")
)

def embed_batch(texts):
    response = client.embeddings.create(
        model = "BAAI/bge-m3",
        input = texts
    )
    return [
        item.embedding
        for item in response.data
    ]

chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

col = chroma.get_or_create_collection(
    "my_notes",
    metadata={"hnsw:space": "cosine"}
)

DISTANCE_THRESHOLD = 0.5

test_cases = [
    ("MCP是什么？", True),
    ("RAG是什么？", True),
    ("什么是长期记忆？", True),
    ("我养的猫叫什么名字？", False),
    ("提拉米苏怎么制作？", False),
    ("恐龙灭绝的陨石坑有多大？", False),
    ("MCP协议是哪一年发布的？", False),
    ("RAG最早由哪家公司提出？", False),
    ("长期记忆最多可以保存多少GB？", False),
]

questions = [
    question
    for question, expected in test_cases
]

q_vecs = embed_batch(questions)

results = col.query(
    query_embeddings=q_vecs,
    n_results=1
)
correct_count = 0
for (question, expected), distance in zip(test_cases, results["distances"]):
    top_distance = distance[0]
    predicted = top_distance <= DISTANCE_THRESHOLD
    is_correct = predicted == expected

    if is_correct:
        correct_count += 1

    print(
        question,
        "距离：", round(top_distance, 3),
        "预期：", expected,
        "判断：", predicted,
        "正确：", is_correct
    )
print("accuracy ：", correct_count / len(test_cases))