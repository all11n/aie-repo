"""Конфигурация приложения."""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

class Settings(BaseSettings):
    APP_NAME: str = "CVD Multi-Label Risk Calculator"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    MODEL_PATH: str = str(ARTIFACTS_DIR / "random_forest_model.pkl")
    SCALER_PATH: str = str(ARTIFACTS_DIR / "scaler.pkl")

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()