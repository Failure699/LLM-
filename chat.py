# -*- coding: utf-8 -*-
# 多轮对话版：能连续聊天的机器人
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 存整个对话历史 —— 大模型不记得你之前说过什么，得我们自己带上
messages = []

print("开始聊天，输入 quit 退出：")
while True:
    user_input = input("你：")            # 等你打字
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})   # 把你的话加进历史

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,                # 把【完整历史】发过去
    )

    reply = response.choices[0].message.content
    print("AI：", reply)

    messages.append({"role": "assistant", "content": reply})    # 把它的回答也加进历史
