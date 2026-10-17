from mcp.server.mcpserver import MCPServer   # ① 导入（已帮你写好）
import os
import requests
from dotenv import load_dotenv
# ② 开店：建一个 server，名字叫 "weather"
load_dotenv()  # 从 .env 读密钥（.env 已被 .gitignore 排除，不会进仓库）
KEY = os.getenv("QWEATHER_API_KEY")
HOST = os.getenv("QWEATHER_API_HOST")

server = MCPServer("weather")  # ② 开店：建一个 server，名字叫 "weather"
# ③ 挂菜单：用装饰器把函数注册成工具（注意：装饰器要紧贴 def，中间别空行）
@server.tool()
# ④ 补 docstring：一句话说这工具干嘛（会变 description）
def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    try:
        # 第 1 步：城市名 → 城市 ID（lang=zh 让天气描述统一返回中文，不随当地语言变）
        geo = requests.get(f"https://{HOST}/geo/v2/city/lookup",
                           params={"location": city, "key": KEY, "lang": "zh"},
                           timeout=10).json()
        # 查不到城市时 code 不是 200，location 也是空列表，给友好提示而不是崩
        if geo.get("code") != "200" or not geo.get("location"):
            return f"查不到城市「{city}」，请检查城市名是否正确"

        # 第 2 步：城市 ID → 实时天气
        resp = requests.get(f"https://{HOST}/v7/weather/now",
                            params={"location": geo["location"][0]["id"], "key": KEY, "lang": "zh"},
                            timeout=10).json()
        if resp.get("code") != "200":
            return f"查询「{city}」天气失败（错误码 {resp.get('code')}）"
        now = resp["now"]
        return f"{city}, {now['temp']}°C, {now['text']}"
    except requests.RequestException:
        # 断网 / 超时 / 域名不通等网络层错误，别让工具直接抛异常
        return f"查询「{city}」天气时网络出错，请稍后再试"

# ⑤ 开门：让 server 跑起来
if __name__ == "__main__":
    server.run()
