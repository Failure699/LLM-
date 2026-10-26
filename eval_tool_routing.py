import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
SYSTEM = """
你是一个带工具的个人助手。

规则：
1. 用户透露关于自己的重要且长期有效的信息时，调用 remember 保存。
2. 回答涉及用户个人信息的问题时，先调用 recall 查询，不能凭空猜测。
3. 查询实时天气时，调用 get_weather。
4. 工具返回结果后，根据结果回答用户。
5. 用户询问技术概念、项目实现、学习进度，或明确要求查询私人笔记时，
   先调用 search_notes。
"""
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
ROUTING_CASES = [
    ("MCP是什么？", {"search_notes"}),
    ("请从我的笔记里查询RAG是什么", {"search_notes"}),
    ("北京天气怎么样？", {"get_weather"}),
    ("我的专业是什么？", {"recall"}),
    ("请记住我的学习方式是先猜再跑", {"remember"}),
    ("1+1等于多少？", set()),
    ("你好", set()),
    ("查询北京天气，同时告诉我是什么专业", {"get_weather", "recall"}),
    ("分别查询北京和上海天气", {"get_weather"}),
]
ROUTING_CASES.extend([
    ("北京是中国的首都吗？", set()),
    ("天气不错", set()),
    ("从我的笔记中查询天气API怎么使用", {"search_notes"}),
    ("北京天气API在我的笔记里是怎么写的？", {"search_notes"}),
    ("请记住我住在南京，然后查询南京天气", {"remember", "get_weather"}),
    ("我叫什么名字，同时查询上海天气", {"recall", "get_weather"}),
])
WEATHER_ARGUMENT_CASES = [
    ("查询南京天气", {"南京"}),
    ("分别查询北京和上海天气", {"北京", "上海"}),
    ("查询广州、深圳和杭州天气", {"广州", "深圳", "杭州"}),
    ("查询北京天气，同时告诉我是什么专业", {"北京"}),
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "在用户的私人笔记中检索与问题相关的资料",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "保存用户透露的重要且长期有效的个人信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "查询已经保存的用户长期记忆",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
        },
    },
]

def check_weather_arguments(predicted_calls, expected_cities):
    weather_calls = [
        call
        for call in predicted_calls
        if call["name"] == "get_weather"
    ]

    actual_cities = {
        call["arguments"].get("city")
        for call in weather_calls
    }

    return (
        len(weather_calls) == len(expected_cities)
        and actual_cities == expected_cities
    )
def predict_tools(question):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        tools=TOOLS,
        temperature=0,
    )

    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []
    predicted = {
        tool_call.function.name
        for tool_call in tool_calls
    }
    predicted_calls = []
    for tool_call in tool_calls:
        predicted_calls.append({
            "name" : tool_call.function.name,
            "arguments" : json.loads(tool_call.function.arguments),
        })
    return predicted, predicted_calls
def run_argument_eval():
    correct_count = 0

    for question, expected_cities in WEATHER_ARGUMENT_CASES:
        _, predicted_calls = predict_tools(question)

        correct = check_weather_arguments(
            predicted_calls,
            expected_cities,
        )

        if correct:
            correct_count += 1

        print(
            "问题：", question,
            "预期城市：", expected_cities,
            "详细调用：", predicted_calls,
            "参数正确：", correct,
        )
    accuracy = correct_count / len(WEATHER_ARGUMENT_CASES)
    print(
        f"天气参数准确率："
        f"{correct_count}/{len(WEATHER_ARGUMENT_CASES)} "
        f"= {accuracy:.1%}"
    )
    return correct_count, len(WEATHER_ARGUMENT_CASES)
def run():
    correct_count = 0

    for question, expect in ROUTING_CASES:
        predicted, predicted_calls = predict_tools(question)

        correct = expect == predicted
        if correct:
            correct_count += 1

        print(
            "问题：", question,
            "预期 ", expect,
            "详细调用：", predicted_calls,
            "路由正确：", correct,
        )

    accuracy = correct_count / len(ROUTING_CASES)
    print(
        f"工具种类路由准确率："
        f"{correct_count}/{len(ROUTING_CASES)} "
        f"= {accuracy:.1%}"
    )
    return correct_count, len(ROUTING_CASES)
REPEAT_COUNT = 3
if __name__ == "__main__":
    total_correct = 0
    total_cases = 0
    weather_correct = 0
    weather_case = 0
    for round_index in range(1, REPEAT_COUNT + 1):
        print(f"\n===== 第 {round_index} 轮路由评测 =====")

        correct_count, case_count = run()
        argument_correct, argument_count = run_argument_eval()
        total_correct += correct_count
        total_cases += case_count
        weather_correct += argument_correct
        weather_case += argument_count

    total_accuracy = total_correct / total_cases
    weather_accuracy =  weather_correct / weather_case

    print(
        f"\n三轮路由总结果："
        f"{total_correct}/{total_cases} "
        f"= {total_accuracy:.1%}"
        f"\n三轮天气参数总结果："
        f"{weather_correct} / {weather_case} "
        f"{weather_accuracy:.1%}"
    )