import asyncio
import os
import json
import copy
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
class AgentState(MessagesState):
    retry_count : int
    critique : str
    review_passed : bool

load_dotenv()
with open(".mcp.json", "r", encoding = "utf-8") as file:
    mcp_config = json.load(file)
server_configs = mcp_config["mcpServers"]
for server_name, server_config in server_configs.items():
    print("Server名称：", server_name)
    print("启动命令：", server_config["command"])
    print("启动参数：", server_config["args"])
MAX_RETRIES = 3
USER_ID = "user_002"
def build_graph(tool_sessions, deepseek_tools, checkpointer):
    model = ChatOpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
    )
    model_with_tools = model.bind_tools(deepseek_tools)
    critic_model = model.bind(
        response_format={"type": "json_object"}
    )
    async def critic(state):
        old_count = state.get("retry_count", 0)
        answer_message = state["messages"][-1]
        answer = answer_message.content

        tool_results = []

        for m in reversed(state["messages"]):
            if m.type == "human":
                question = m.content
                break

            if m.type == "tool":
                tool_results.append(m.content)

        tool_results.reverse()
        if tool_results:
            tool_evidence = "\n".join(tool_results)
        else:
            tool_evidence = "本轮没有调用工具"
        history_lines = []
        for history_message in state["messages"][:-1]:
            if history_message.type == "human":
                role = "用户"
            elif history_message.type == "ai":
                role = "助手"
            elif history_message.type == "tool":
                role = "工具"
            else:
                continue

            if history_message.content:
                history_lines.append(
                    f"{role}：{history_message.content}"
                )

        history_text = "\n".join(history_lines[-10:])

        if not history_text:
            history_text = "没有更早的对话历史"
        message = [
            SystemMessage("""你是严格评审，逐条检查回答，两条都要看：
            ① 正确：回答内容本身对不对，有没有瞎编。
            ② 有用：回答对用户有没有实际帮助。如果用户的问题本身就没意义
            （比如只发一个数字、一个表情、无意义字符），或者回答只是敷衍、
            复读、绕圈子，就属于「没用」。
            只要有一条不满足，就判「错误」。
            必须只输出合法 JSON，不要输出 Markdown 或其他文字。
            JSON 格式示例：
            {
            "passed": true,
            "reason": "回答与工具证据一致"
            }。
            如果提供了工具证据，以工具证据为事实依据；
            回答与工具证据一致时，不要以“缺少数据来源”为由判错。
            如果回答涉及“刚才”“上一条”“之前聊过什么”等对话上下文，
            应以最近对话历史为事实依据，不能因为本轮没有调用工具就判错。"""),
            HumanMessage(f"最近对话历史：\n{history_text}\n\n"
            f"当前问题：{question}\n\n"
            f"本轮工具证据：{tool_evidence}\n\n"
            f"待评审回答：{answer}")
        ]
        review = await critic_model.ainvoke(message)
        try:
            review_data = json.loads(review.content)
            passed = review_data["passed"]
            reason = review_data["reason"]

            if not isinstance(passed, bool):
                raise TypeError("passed必须是布尔值")
        except (json.JSONDecodeError, KeyError, TypeError):
            passed = False
            reason = "critic返回的JSON格式异常"
        if not passed:
            new_count = old_count + 1
            message_update = [
                RemoveMessage(id = answer_message.id)
            ]
        else:
            new_count = 0
            message_update = []
        #print(review.content)
        print(
        "评审：", reason,
        "是否通过：", passed,
        "retry_count：", new_count
        )
        return {
                "critique" : reason,
                "retry_count" : new_count,
                "messages" : message_update,
                "review_passed" : passed
                }
    async def call_model(state):
        passed = state.get("review_passed", False)
        critique = state.get("critique", "")
        messages = [
            SystemMessage(
                """你是一个带工具的个人助手。
                1. 用户透露长期有效的个人信息时，调用 remember。
                2. 回答用户个人信息时，先调用 recall。
                3. 查询实时天气时，调用 get_weather。
                4. 查询技术概念、项目实现和学习笔记时，调用 search_notes。
                5. 工具返回结果后，根据工具证据回答，不能编造。"""
            )
        ] + state["messages"]
        if not passed and critique:
            messages.append(
                HumanMessage( f"上一版回答的评审意见：{critique}\n"
                    "请根据意见重新回答用户原问题。" )
            )
        response = await model_with_tools.ainvoke(messages)
        return {
            "messages" : [response]
        }
    async def execute_mcp_tool(tool_call):
        try:
            tool_name = tool_call["name"]
            tool_args = dict(tool_call["args"])
            session = tool_sessions.get(tool_name)
            if session is None:
                return f"找不到工具对应的MCP Server：{tool_name}"
            if tool_name in ("remember", "recall"):
                tool_args["user_id"] = USER_ID
            tool_result = await session.call_tool(
                tool_name,
                arguments = tool_args
            )
        except Exception as error:
            return f"工具调用失败： {error}"
        tool_texts = []
        for item in tool_result.content:
            if item.type == "text":
                tool_texts.append(item.text)
        tool_texts = "\n".join(tool_texts)
        if tool_result.is_error:
            return f"工具返回错误： {tool_texts}"
        if not tool_texts:
            return "工具没返回文字结果"
        return tool_texts
    async def call_mcp_tools(state):
        last_message = state["messages"][- 1]
        tool_calls = last_message.tool_calls
        print(
            "本轮调用工具：",
            [tool_call["name"] for tool_call in tool_calls]
        )
        tool_tasks = []
        for tool_call in tool_calls:
            task = execute_mcp_tool(tool_call)
            tool_tasks.append(task)
        tool_results = await asyncio.gather(
            *tool_tasks,
            return_exceptions = True
        )
        tool_messages = []
        for tool_call, tool_result in zip(
            tool_calls,
            tool_results
        ):
            if isinstance(tool_result, Exception):
                tool_result = f"工具调用失败： {tool_result}"
            tool_message = ToolMessage(
                content = str(tool_result),
                tool_call_id = tool_call["id"],
                name = tool_call["name"]
            )
            tool_messages.append(tool_message)
        return {
            "messages" : tool_messages
        }
    def route_after_model(state):
        last_message = state["messages"][- 1]
        if last_message.tool_calls:
            return "mcp_tools"
        return "critic"
    def route_after_critic(state):
        retry_count = state.get("retry_count", 0)
        passed = state.get("review_passed", False)
        if passed or retry_count >= MAX_RETRIES:
            return END
        return  "model"
    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node("mcp_tools", call_mcp_tools)
    builder.add_node("critic", critic)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {
            "mcp_tools" : "mcp_tools",
            "critic" : "critic"
        }
    )
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "model" : "model",
            END : END
        }
    )
    builder.add_edge("mcp_tools", "model")
    app = builder.compile(
        checkpointer=checkpointer
    )
    return app
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
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(
                "checkpoints_10.sqlite"
            )
        )
        app = build_graph(
            tool_sessions,
            deepseek_tools,
            checkpointer
        )
        while True:
            ques = input("你的问题是：")
            if ques.strip() in ("quit", "exit", "q", "退出"):
                break
            if not ques.strip():
                print("请输入问题")
                continue
            try:
                result = await app.ainvoke(
                    {
                        "messages" : [
                            HumanMessage(content = ques)
                        ],
                        "retry_count" : 0,
                        "critique" : "",
                        "review_passed" : False
                    },
                    config = {
                        "configurable": {
                            "thread_id": "user_002_chat_2"
                        },
                        "recursion_limit" : 20
                    }
                )
            except Exception as error:
                print("Agent运行失败 ", error)
                continue
            if result.get("review_passed", False):
                final_message = result["messages"][- 1]
                print("最终回答： ", final_message.content)
            else :
                print("太难了，我不会")


if __name__ == "__main__":
    asyncio.run(main())
