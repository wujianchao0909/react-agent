from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="用户输入")
    thread_id: str = Field(..., min_length=1, max_length=128, description="会话 ID，前端生成并持久化")

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    version: str = "0.1.0"