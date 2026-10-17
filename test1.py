import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# ============ 实验 B：temperature 的随机性 ============
print("\n========== B1：temperature=0，问两次 ==========")
for _ in range(2):
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "用一句话形容夕阳"}],
        temperature=0,
    )
    print("·", r.choices[0].message.content)

print("\n========== B2：temperature=1.5，问两次 ==========")
for _ in range(2):
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "用一句话形容夕阳"}],
        temperature=2,
    )
    print("·", r.choices[0].message.content)