import asyncio
import os
import json
import copy
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
load_dotenv()
with open(".mcp.json", "r", encoding = "utf-8") as file:
    mcp_config = json.load(file)
server_configs = mcp_config["mcpServers"]
for server_name, server_config in server_configs.items():
    print("Server名称：", server_name)
    print("启动命令：", server_config["command"])
    print("启动参数：", server_config["args"])
MAX = 5
USER_ID = "user_002"
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
weather_server_params = StdioServerParameters(
    command = ".venv/Scripts/python.exe",
    args = ["mcp_weather.py"]
)
memory_server_params = StdioServerParameters(
    command=".venv/Scripts/python.exe",
    args=["mcp_memory.py"]
)
async def execute_tool(tool_sessions, tool_call, user_id):
    try:
        tool_name = tool_call.function.name
        session = tool_sessions.get(tool_name)
        if session is None:
            return f"找不到工具对应的 MCP Server：{tool_name}"
        tool_args = json.loads(tool_call.function.arguments)
        if tool_name in ("remember", "recall"):
            tool_args["user_id"] = user_id
        print("实际发送给 MCP 的参数：", tool_args)
        tool_result = await session.call_tool(
            tool_name,
            arguments = tool_args
        )
    except json.JSONDecodeError:
        return "工具json不合法"
    except Exception as error:
        return f"工具调用失败 {error}"
    tool_text = []
    for item in tool_result.content:
        if item.type == "text":
            tool_text.append(item.text)
    tool_text = "\n".join(tool_text)
    if tool_result.is_error:
            return f"工具返回错误：{tool_text}"
    if not tool_text:
            return "工具没有返回文字结果"
    return tool_text
async def main():
     async with AsyncExitStack() as stack:
        sessions = {}
        for server_name, server_config in server_configs.items():
            server_params = StdioServerParameters(
                command = server_config["command"],
                args = server_config.get("args", []),
                env = server_config.get("env") or None
            )
            read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server_params)
            )
            session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            sessions[server_name] = session
            print("已连接Server： ", server_name)
        tool_sessions = {}
        deepseek_tools = []
        for server_name, session in sessions.items():
            tool_result = await session.list_tools()
            for tool in tool_result.tools:
                if tool.name in tool_sessions:
                    raise ValueError(f"发现重命名工具: {tool.name}")
                tool_sessions[tool.name] = session
                parameters = copy.deepcopy(tool.input_schema)
                if (tool.name in ("remember", "recall")):
                    parameters.get("properties", {}).pop("user_id", None)
                    if "required" in parameters:
                        parameters["required"] = [
                            name
                            for name in parameters["required"]
                            if name != "user_id"
                        ]
                deepseek_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": parameters
                        }
                    })
                print(
                    f"已接入工具:  {server_name}.{tool.name}"
                    )
        messages = [
                    {
                        "role": "system",
                            "content":
                        """你是一个带工具的个人助手。
                        规则：
                        1. 用户透露关于自己的重要且长期有效的信息时，调用 remember 保存。
                        2. 回答涉及用户个人信息的问题时，先调用 recall 查询，不能凭空猜测。
                        3. 查询实时天气时，调用 get_weather。
                        4. 工具返回结果后，根据结果回答用户。    """
                    }
                ]
        while True:
            ques = input("请输入你的问题： ")
            if ques.strip() in ("quit", "exit", "q", "退出"):
                break
            if not ques.strip():
                print("请输入具体问题")
                continue
            turn_start = len(messages)
            messages.append({"role": "user", "content": ques})
            step_count = 0
            while step_count < MAX:
                step_count += 1
                print(f"\n===== Agent 第 {step_count} 轮 =====")
                try:
                    if step_count == MAX:
                        # 第5次：不提供工具，只能整理最终答案
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages
                        )
                        messages.append(response.choices[0].message)
                        print("达到最大工具轮数，开始强制生成最终回答")
                        print("最终回答是:" ,response.choices[0].message.content)
                        break
                    response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            tools=deepseek_tools
                    )
                except Exception as error:
                    print("DeepSeek请求失败：", error)
                    # 请求失败时回滚本轮消息，避免残缺对话污染后续上下文。
                    del messages[turn_start:]
                    break
                assistant_message = response.choices[0].message
                print("模型文字：", assistant_message.content)
                print("模型要调用的工具：", assistant_message.tool_calls)
                if assistant_message.tool_calls:
                    messages.append(assistant_message)
                    tool_tasks = []

                    for tool_call in assistant_message.tool_calls:
                        task = execute_tool(tool_sessions, tool_call, USER_ID)
                        tool_tasks.append(task)
                    tool_texts = await asyncio.gather(*tool_tasks, return_exceptions= True )
                    for tool_call, tool_text in zip(
                        assistant_message.tool_calls,
                        tool_texts
                    ):
                        if isinstance(tool_text, Exception):
                            tool_text = f"工具调用失败：{tool_text}"

                        print("工具调用：", tool_call.function.name)
                        #print("整理后的工具结果：", tool_text)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_text
                        })
                else:
                    print("最终回答：", assistant_message.content)
                    messages.append(assistant_message)
                    break

            
if __name__ == "__main__":
    asyncio.run(main())
