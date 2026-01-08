# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-03.csv`
- Размер: (15000, 30)
- Целевая переменная: `target` (0 - 8138, 1 - 4535, 2 - 2327)
- Признаки: float64, int64 для id и target

## 2. Protocol

- Разбиение: train/test (test_size = 0.2, `random_state` = 42)
- Подбор: CV на train (2-3 фолда, оптимизировала их количество для уменьшения времени обучения)
- Метрики: accuracy, F1 макро, ROC-AUC (accuracy интуитивно понятна, f1 уместна для мультиклассовой классификации и макро-усреднение дает равный вес каждому классу, ROC-AUC OVR показывает, насколько хорошо модель ранжирует примеры и устойчив к дисбалансу)

## 3. Models

Опишите, какие модели сравнивали и какие гиперпараметры подбирали.

Минимум:

- DummyClassifier (baseline, stratified)
- LogisticRegression (baseline, max_iter=2000, random_state=42)
- DecisionTreeClassifier (параметры подбирала при помощь GridSearchCV max_depth: [3, 5, 7, 10, 15, None], min_samples_leaf: [1, 2, 4, 8, 16], ccp_alpha: [0.0, 0.001, 0.01, 0.1])
- RandomForestClassifier (параметры подбирала при помощь GridSearchCV n_estimators: [50, 100, 200], max_depth: [10, 20, 30, None], min_samples_leaf: [1, 2, 4], max_features: ['sqrt'])
- GradientBoosting(параметры подбирала при помощь GridSearchCV n_estimators: [50, 100, 150], learning_rate: [0.01, 0.05, 0.1, 0.2], max_depth: [3, 5, 7], min_samples_split: [2, 5, 10], subsample: [0.8, 0.9, 1.0])

Опционально:

- StackingClassifier (с CV-логикой)

## 4. Results

- Таблица/список финальных метрик на test по всем моделям
Модель	Accuracy	F1-score (macro)	ROC-AUC (OVR)
DummyClassifier	0.3345	0.3338	0.5000
LogisticRegression	0.4517	0.4483	0.8102
DecisionTree	0.6483	0.6450	0.8349
RandomForest	0.7586	0.7552	0.9408
GradientBoosting	0.7414	0.7363	0.9271

- Победитель (по ROC-AUC или по согласованному критерию) RandomForestClassifier (максимальный F1-score 0.7552 и ROC-AUC 0.9408)
Лучший баланс метрик, плюс ансамбль снижает variance

## 5. Analysis

- Устойчивость: что будет, если поменять `random_state` (хотя бы 5 прогонов для 1-2 моделей) – качество предсказуемо, получается что при любом random_state модели показывают схожие результаты
- Ошибки: confusion matrix для лучшей модели (классы 1 и 2 чаще путаются между собой, класс 0 более отличим от остальных)
- Интерпретация: permutation importance (top-10/15) - неравномерное распределение важности, feature_13 и feature_5 - наиболее важные признаки

## 6. Conclusion

Контроль сложности обязателен, так как без регуляризации DecisionTree переобучается
RandomForest улучшил DecisionTree на 20% по F1-score
RandomForest показал себя лучше GradientBoosting на этих данных
max_features='sqrt' дал лучший результат
