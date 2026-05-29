"""Основное приложение FastAPI."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.service.endpoints import router
from src.service.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Мультилейбловая классификация сердечно-сосудистых заболеваний",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутера
app.include_router(router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "CVD Multi-Label Risk Calculator API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict"
    }


@app.on_event("startup")
async def startup_event():
    """Событие при запуске сервера."""
    logger.info("Server starting...")
    logger.info(f"App: {settings.APP_NAME} v{settings.VERSION}")


@app.on_event("shutdown")
async def shutdown_event():
    """Событие при остановке сервера."""
    logger.info("Server shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.service.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )