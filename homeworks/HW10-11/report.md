# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Часть A: Выбран датасет STL10 для классификации изображений. Этот датасет содержит 10 классов объектов естественных сцен, имеет достаточно высокое разрешение изображений (96x96), что позволяет продемонстрировать работу CNN и transfer learning.
- Часть B: Выбран датасет Pascal VOC с треком detection. Использована pretrained модель Faster R-CNN для детекции объектов. Этот трек выбран как наиболее универсальный и наглядный для демонстрации работы detection-пайплайна.
- Сравнения: В части A сравнивались 4 эксперимента (C1-C4): простая CNN без аугментаций, простая CNN с аугментациями, ResNet18 с замороженным backbone, ResNet18 с partial fine-tuning. Во второй части сравнивались 2 режима инференса с разными порогами уверенности (V1: 0.3, V2: 0.7).

## 2. Среда и воспроизводимость

- Python: 3.10+
- torch / torchvision: 2.0+ / 0.15+
- Устройство (CPU/GPU): CUDA (при наличии) / CPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение: train/val/test  (80/20 для val из train, официальный test)
- Базовые transforms: Resize(96x96) → ToTensor → Normalize(0.5, 0.5)
- Augmentation transforms: Resize → RandomHorizontalFlip → RandomRotation(10) → ColorJitter → ToTensor → Normalize
- Комментарий: STL10 содержит 10 классов (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). Изображения имеют размер 96x96, что достаточно для обучения CNN. Датасет не имеет официального validation split, поэтому val был отделён от train с фиксированным seed для воспроизводимости.

### 3.2. Часть B: structured vision

- Датасет: Pascal VOC
- Трек: detection
- Что считается ground truth: Bounding boxes с классами объектов из COCO (используются в pretrained модели)
- Какие предсказания использовались: Предсказания Faster R-CNN с фильтрацией по score threshold
- Комментарий: Detection трек выбран как наиболее наглядный для демонстрации работы pretrained моделей. Pascal VOC содержит размеченные bounding boxes для множественных объектов на изображении, что позволяет корректно оценить precision/recall при разных порогах уверенности.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): Простая CNN с 3 сверточными слоями (32→64→128 фильтров) + FC слой. Обучается без аугментаций.
- C2 (simple-cnn-aug): Та же архитектура CNN, но обучение с аугментациями (flip, rotation, color jitter).
- C3 (resnet18-head-only): ResNet18 с pretrained weights на ImageNet. Backbone заморожен, обучается только классификационная голова (fc слой).
- C4 (resnet18-finetune): ResNet18 с pretrained weights. Backbone частично разморожен (layer4 + fc), fine-tuning с разными learning rate.

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (lr=0.001 для C1-C3, lr=0.0001/0.001 для C4)
- Batch size: 32
- Epochs (макс): 10
- Критерий выбора лучшей модели: best_val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track
- Датасет Pascal VOC.
- Трек detection. Этот трек выбран как наиболее универсальный и наглядный для демонстрации работы detection-пайплайна.
- Модель: FasterRCNN_ResNet50_FPN (pretrained на COCO)
- V1: `score_threshold = 0.3` более низкий порог, больше детекций
- V2: `score_threshold = 0.7` более высокий порог, меньше детекций
- Как считался IoU: Базовое сопоставление prediction ↔ ground truth при IoU >= 0.5
- Как считались precision / recall: Precision = TP/(TP+FP), Recall = TP/(TP+FN) на основе сопоставленных детекций

### Если выбран segmentation track

- Модель:
- Что считается foreground:
- V1: базовая постобработка
- V2: альтернативная постобработка
- Как считался mean IoU:
- Считались ли дополнительные pixel-level метрики:

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples.png`
- Метрики detection (precision/recall/IoU): `./artifacts/figures/detection_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.87
- Итоговая `test_accuracy` лучшего классификатора: 0.95
- Что дали аугментации (C2 vs C1): Улучшение generalization, снижение overfitting на train
- Что дал transfer learning (C3/C4 vs C1/C2): Значительное улучшение accuracy благодаря pretrained features
- Что оказалось лучше: head-only или partial fine-tuning: Partial fine-tuning (C4) обычно показывает лучшую accuracy
- Что показал режим V1 во второй части:  Больше детекций, выше recall, ниже precision
- Что показал режим V2 во второй части: Меньше детекций, выше precision, ниже recall
- Как интерпретируются метрики второй части: При увеличении порога модель становится более консервативной, отфильтровывая неуверенные предсказания

## 7. Анализ

Простая CNN (C1/C2) показывает ограниченные результаты на STL10 из-за малого количества данных и относительно простой архитектуры. Модель быстро переобучается без регуляризации.
Аугментации (C2) дали устойчивое улучшение validation accuracy по сравнению с базовой версией (C1). RandomHorizontalFlip и ColorJitter помогли модели лучше обобщать на новые данные.
Pretrained ResNet18 (C3/C4) значительно превзошёл простую CNN благодаря features, обученным на ImageNet. Это подтверждает эффективность transfer learning для задач с ограниченными данными.
Partial fine-tuning (C4) показал лучшие результаты чем head-only (C3), так как адаптация layer4 позволила модели лучше настроиться на специфику STL10.
Для detection задачи выбранные метрики (precision/recall/IoU) подходят под задачу оценки качества детекции объектов. Они позволяют оценить компромисс между количеством найденных объектов и точностью предсказаний.
При переходе от V1 к V2 (увеличение порога с 0.3 до 0.7) precision увеличился, а recall уменьшился. Это ожидаемое поведение: модель отфильтровывает неуверенные предсказания.
Наиболее показательными оказались ошибки на мелких объектах и объектах с частичной окклюзией — pretrained модель иногда пропускает их или предсказывает с низкой уверенностью.

## 8. Итоговый вывод

В качестве базового конфига классификации я бы взял C4 (ResNet18 + partial fine-tuning), так как он даёт лучший баланс между accuracy и временем обучения.
Главное, что я понял про transfer learning: pretrained веса на больших датасетах (ImageNet) позволяют достичь хороших результатов даже на небольших датасетах с минимальным дообучением.
Главное, что я понял про detection/segmentation: выбор порога уверенности критически влияет на precision/recall trade-off, и оптимальный порог зависит от конкретной задачи (лучше найти всё или лучше найти только уверенное).

## 9. Приложение (опционально)

Если вы делали дополнительные сравнения:

Дополнительные fine-tuning сценарии: не проводились
Confusion matrix для классификации: не включена в базовую версию
Дополнительная постобработка для второй части: не проводилась
Дополнительные графики: ./artifacts/figures/data_samples.png
