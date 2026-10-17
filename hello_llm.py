# -*- coding: utf-8 -*-
# 第一个 LLM 脚本：调 DeepSeek 大模型
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. 创建"客户端" —— 相当于拿到大模型的遥控器
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从 .env 读，别写死
    base_url="https://api.deepseek.com",            # DeepSeek 的接口地址
)

# 2. 发消息、拿回复
response = client.chat.completions.create(
    model="deepseek-chat",                          # 用哪个模型
    messages=[
        {"role": "user", "content": "printfhello world"},
    ],
)

# 3. 打印大模型的回答
print(response.choices[0].message.content)
