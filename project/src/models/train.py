"""Модуль для обучения и оценки моделей."""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """
    Обучает модель и возвращает метрики на тестовой выборке.
    
    Args:
        X: Признаки
        y: Целевые переменные (мультилейбл)
        model_type: Тип модели ("random_forest" или "logistic")
        test_size: Размер тестовой выборки
        random_state: Seed для воспроизводимости
        
    Returns:
        model: Обученная модель
        metrics: Словарь с метриками
    """
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Выбор модели
    if model_type == "random_forest":
        base_clf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "logistic":
        base_clf = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Оборачиваем в MultiOutputClassifier для мультилейбловой классификации
    model = MultiOutputClassifier(base_clf)
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    metrics = evaluate_model(y_test, y_pred)
    
    return model, metrics


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики качества для мультилейбловой классификации.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        
    Returns:
        metrics: Словарь с метриками
    """
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }
    
    return metrics


def get_model_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Получает предсказания модели.
    
    Args:
        model: Обученная модель
        X: Признаки
        
    Returns:
        y_pred: Предсказания
    """
    return model.predict(X)