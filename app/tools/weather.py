import httpx
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

from app.config import settings

@tool
async def get_weather_real(city:Annotated[str, Field(description="城市名称，如 北京、上海")]) -> str:
    """查询指定城市的实时天气。当用户询问天气状况时使用。"""
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        try:
            geo_resp = await client.get(settings.qweather_geo_url,
                                    params={"location": city, "key": settings.qweather_api_key}
                                    )

            geo_data = geo_resp.json()

            if geo_data.get("code") != "200" or not geo_data.get("location"):
                return f"未找到城市：{city}"

            city_id = geo_data["location"][0]["id"]

            weather_resp = await client.get(settings.qweather_now_url,
                                    params={"location": city_id, "key": settings.qweather_api_key},
                                    )

            weather_data = weather_resp.json()

            if weather_data.get("code") != "200":
                return f"天气查询失败，错误码：{weather_data.get('code')}"

            now = weather_data.get("now")

            if not now:
                return "天气数据为空"

            return (f"{city}当前天气：{now['text']}，"
                    f"温度{now['temp']}°C，"
                    f"体感{now['feelsLike']}°C，"
                    f"湿度{now['humidity']}%")
        except httpx.TimeoutException:
            return f"天气查询超时：{city}"
        except httpx.HTTPError as e:
            return f"天气查询网络错误：{e}"
        except ValueError as e:
            return f"天气接口返回非 JSON：{e}"
        except Exception as e:
            return f"天气查询未知错误：{type(e).__name__}: {e}"
