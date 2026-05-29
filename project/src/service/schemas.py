"""Pydantic схемы для API."""
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class PredictRequest(BaseModel):
    """Входные данные для предсказания. Должны совпадать с признаками модели."""
    age: int = Field(..., ge=18, le=100, description="Возраст (лет)")
    gender: int = Field(..., ge=1, le=2, description="Пол (1 - женский, 2 - мужской)")
    height: float = Field(..., ge=100, le=250, description="Рост (см)")
    weight: float = Field(..., ge=30, le=200, description="Вес (кг)")
    ap_hi: int = Field(..., ge=80, le=200, description="Систолическое давление (мм рт.ст.)")
    ap_lo: int = Field(..., ge=40, le=130, description="Диастолическое давление (мм рт.ст.)")
    cholesterol: int = Field(..., ge=1, le=3, description="Холестерин (1-норма, 2-выше, 3-высокий)")
    gluc: int = Field(..., ge=1, le=3, description="Глюкоза (1-норма, 2-выше, 3-высокий)")
    smoke: int = Field(..., ge=0, le=1, description="Курение (0 - нет, 1 - да)")
    alco: int = Field(..., ge=0, le=1, description="Алкоголь (0 - нет, 1 - да)")
    active: int = Field(..., ge=0, le=1, description="Физ. активность (0 - нет, 1 - да)")

class DiseasePrediction(BaseModel):
    """Предсказание по одному заболеванию."""
    disease: str
    probability: float
    risk_level: str  # "Низкий", "Средний", "Высокий"

class PredictResponse(BaseModel):
    """Ответ API с предсказаниями."""
    predictions: List[DiseasePrediction]
    timestamp: datetime
    model_version: str

class HealthResponse(BaseModel):
    """Статус сервиса."""
    status: str
    model_loaded: bool
    timestamp: datetime