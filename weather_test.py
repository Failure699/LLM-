# weather_test.py —— 用真实 API 查天气（先单独跑通，再搬进 mcp_weather.py）
import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("QWEATHER_API_KEY")
HOST = os.getenv("QWEATHER_API_HOST")

def get_weather(city: str) -> str:
    """城市名 → 城市 ID → 实时天气。返回如 '北京, 15°C, 小雨'"""
    # ── 第 1 步：城市名 → 城市 ID ──
    # 调 GET https://{HOST}/v7/geo/city/lookup?location=城市名&key=KEY
    # TODO①：用 requests.get(url, params={...}).json() 把返回存进 geo
    #geo = requests.get(f"https://{HOST}/v7/geo/city/lookup", params={"location": city, "key": KEY}).json()
    r = requests.get(f"https://{HOST}/geo/v2/city/lookup", params={"location": city, "key": KEY})
    geo = r.json()
    city_id = geo["location"][0]["id"]

    # ── 第 2 步：城市 ID → 实时天气 ──
    # 调 GET https://{HOST}/v7/weather/now?location=城市ID&key=KEY
    # TODO②：取返回里的 "now"，存进 now
    now = requests.get(f"https://{HOST}/v7/weather/now", params={"location": city_id, "key": KEY}).json()["now"]
    return f"{city}, {now['temp']}°C, {now['text']}"

if __name__ == "__main__":
    print(get_weather("马德里"))
