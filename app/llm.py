from langchain_deepseek import ChatDeepSeek
from app.config import settings

llm = ChatDeepSeek(
    model=settings.deepseek_model,
    api_key=settings.deepseek_api_key,
    api_base=settings.deepseek_base_url,
    temperature=0.01,
    extra_body={"thinking":{"type":"disabled"}}
)