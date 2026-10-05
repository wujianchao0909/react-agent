from fastapi import APIRouter
from app.api.schemas import HealthResponse
from app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(model=settings.deepseek_model)