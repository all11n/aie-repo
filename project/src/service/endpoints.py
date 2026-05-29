"""API эндпоинты."""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from src.service.schemas import PredictRequest, PredictResponse, HealthResponse
from src.models.predictor import MultiLabelPredictor

logger = logging.getLogger(__name__)
router = APIRouter()

def get_predictor() -> MultiLabelPredictor:
    """Зависимость для получения предиктора (singleton на время запроса)."""
    return MultiLabelPredictor()

@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    predictor: MultiLabelPredictor = Depends(get_predictor)
):
    """Мультилейбловое предсказание риска 5 заболеваний."""
    logger.info(f"Prediction request: age={request.age}, gender={request.gender}")
    
    try:
        predictions = predictor.predict(request.model_dump())
        logger.info(f"✅ Predictions generated successfully.")
        
        return PredictResponse(
            predictions=predictions,
            timestamp=datetime.now(),
            model_version="trained_v1.0" if not predictor.is_dummy else "dummy_v0.1"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при выполнении предсказания")

@router.get("/health", response_model=HealthResponse)
async def health_check(
    predictor: MultiLabelPredictor = Depends(get_predictor)
):
    """Проверка доступности сервиса и статуса модели."""
    logger.info(" Health check requested")
    return HealthResponse(
        status="healthy",
        model_loaded=not predictor.is_dummy,
        timestamp=datetime.now()
    )