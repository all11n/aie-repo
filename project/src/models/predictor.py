"""Модуль для инференса модели."""
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.data.preprocessing import FEATURE_NAMES, DISEASE_LABELS
from src.service.config import settings
from src.service.schemas import DiseasePrediction

logger = logging.getLogger(__name__)

class MultiLabelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_dummy = False
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = Path(settings.MODEL_PATH)
        scaler_path = Path(settings.SCALER_PATH)

        if model_path.exists() and scaler_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Model & Scaler loaded successfully from {model_path.parent}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.is_dummy = True
        else:
            logger.warning("Model files not found. Running in DUMMY mode.")
            self.is_dummy = True

    def predict(self, input_data: Dict[str, Any]) -> List[DiseasePrediction]:
        if self.is_dummy:
            return self._dummy_predict()

        try:
            X = np.array([[input_data.get(feat, 0) for feat in FEATURE_NAMES]])

            X_scaled = self.scaler.transform(X)
            
            if isinstance(self.model, list):
                probas = [est.predict_proba(X_scaled)[0][1] for est in self.model]
            else:
                raw_probas = self.model.predict_proba(X_scaled)
                probas = [p[0, 1] for p in raw_probas]

            predictions = []
            for i, disease in enumerate(DISEASE_LABELS):
                prob = float(probas[i])
                risk = "Низкий" if prob < 0.3 else ("Средний" if prob < 0.6 else "Высокий")
                predictions.append(DiseasePrediction(
                    disease=disease.replace("_", " ").title(),
                    probability=round(prob, 4),
                    risk_level=risk
                ))
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError("Ошибка инференса модели") from e

    def _dummy_predict(self) -> List[DiseasePrediction]:
        """Заглушка для демонстрации, если модель не найдена."""
        np.random.seed(42)
        return [
            DiseasePrediction(
                disease=d.replace("_", " ").title(),
                probability=round(float(np.random.uniform(0.05, 0.35)), 4),
                risk_level="Низкий"
            ) for d in DISEASE_LABELS
        ]