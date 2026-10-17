
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
      api_key=os.getenv("DEEPSEEK_API_KEY"),                 # 从 .env 读，别写死
      base_url="https://api.deepseek.com",  # ② 关键：指到 DeepSeek
  )

  # ③ 定义工具（格式 = OpenAI 标准，将来在 Claude 里同构）
tools = [
      {
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "查询某个城市的天气",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "city": {"type": "string", "description": "城市名"}
                  },
                  "required": ["city"],
              },
          },
      },
      {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "计算一个数学表达式，例如 3*(2+5)",
                "parameters":
                {
                    "type": "object",
                    "properties":
                    {
                        "a" : {"type" : "string" , "description" : "要计算的式子例如 3*(2+5)"}

                    },
                    "required": ["a"],
                },
            },
      }
  ]
def calculate(a: str) -> str:
    return str(eval(a))  # 这里可以接真实计算 API

  # ④ 你真正执行的函数（模型只"点菜"，你"做菜"）
def get_weather(city: str) -> str:
      return f"beijing: {city}, 40°C, 晴"  # 这里可以接真实天气 API

messages = [{"role": "system", "content": "你是我的仆人，你要以对主人的方式回答我"},
    {"role": "user", "content": "帮我计算一下1 + 2 * 5，顺便北京的天气咋样"}]  # 先给模型一个问题

  # ⑤ 核心循环：调用 → 执行 → 回传 → 再调用
while True:
      response = client.chat.completions.create(
          model="deepseek-chat",
          messages=messages,
          tools=tools,
      )
      msg = response.choices[0].message

      # 模型还想调工具吗？
      if msg.tool_calls:
          messages.append(msg)   # 把模型这轮的话（含 tool_calls）原样塞回历史
          print("→ 走了【调工具】分支")
          for tc in msg.tool_calls:
              print("🔧 模型要调:", tc.function.name, "参数:", tc.function.arguments)  # 加这行
              args = json.loads(tc.function.arguments)  # arguments 是 JSON 字符串，要解析
              if tc.function.name == "get_weather":
                  result = get_weather(args["city"])
              elif tc.function.name == "calculate":
                  result = calculate(args["a"])
              else:
                  result = "未知工具"
              print("📥 函数返回:", result)   # 加这行

              # 每个 tool_call 回一条 role:"tool" 消息
              messages.append({
                  "role": "tool",
                  "tool_call_id": tc.id,   # 用 id 把结果和"哪个调用"对应
                  "content": result,
              })
          
      else:
          print("→ 走了【直接回答】分支")
          print(msg.content)   # ⑥ 模型给出最终答案
          break