from mcp.server.mcpserver import MCPServer   # ① 导入（已帮你写好）

# ② 开店：建一个 server，名字叫 "weather"
server = MCPServer("weather")  # ② 开店：建一个 server，名字叫 "weather"
# ③ 挂菜单：用装饰器把函数注册成工具（注意：装饰器要紧贴 def，中间别空行）
@server.tool()
# ④ 补 docstring：一句话说这工具干嘛（会变 description）
def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    return f"{city}, 40°C, 晴"

# ⑤ 开门：让 server 跑起来
if __name__ == "__main__":
    server.run()
