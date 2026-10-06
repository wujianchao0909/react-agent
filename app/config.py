from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM
    deepseek_api_key: str = "sk-b0096fd449e24084845d075596219fdf"
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Weather
    qweather_api_key: str = "a8221c922b064f26b4b83919d8bfc8ca"
    qweather_geo_url: str = "https://mx4t2dwqk4.re.qweatherapi.com/geo/v2/city/lookup"
    qweather_now_url: str = "https://mx4t2dwqk4.re.qweatherapi.com/v7/weather/now"
    http_timeout: float = 5.0

    # RAG
    embedding_path: str = str(BASE_DIR / "Qwen3-Embedding-0.6B")
    reranker_path: str = str(BASE_DIR / "Qwen3-Reranker-0.6B")
    chroma_dir: str = str(BASE_DIR / "chroma_db")
    docs_dir: str = str(BASE_DIR / "docs")
    retrieval_k: int = 5
    distance_threshold: float = 1.0

    # Memory
    sqlite_path: str = str(BASE_DIR / "conversation.db")

settings = Settings()