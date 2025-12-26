# CAPTCHA-Recognition-CRNN

## Постановка задачи
Распознавание капчи по картинке. Рассматриваются многобуквенные капчи. Используется решение с kaggle. https://www.kaggle.com/code/lapl04/pytorch-captcha-recognizer/notebook#Save

### Формат входных и выходных данных
На вход подаётся фото капчи, на выход - текстовый вид данной капчи

### Метрики
Используемыми метриками в данном решении являются accuracy и hard-accuracy.
Accuracy считаем как доля правильно распознанных в батче символов, а hard-accuracy - доля полностью верно распознанных капч. Данные метрики хорошо отражают цель данной задачи.

### Валидация и тест
Будем использовать 5 различных датасета. 90% используется под train. 10% - под валидацию. Также в предложенном решении в конце происходит тест на валидационных данных, что надо переделать. Думаю, что стоит взять 80% - train, 10% - validation, 10% - test.  Для воспроизводимости фиксируем seed. Воспроизводимость важна для анализа полученного решения и возможности дебажиться.

### Датасеты
**Ссылки** : 

    1. https://www.kaggle.com/datasets/akashguna/large-captcha-dataset
    2. www.kaggle.com/datasets/parsasam/captcha-dataset
    3. https://www.kaggle.com/datasets/aadhavvignesh/captcha-images
    4. https://www.kaggle.com/datasets/fournierp/captcha-version-2-images
    5. https://www.kaggle.com/datasets/jassoncarvalho/comprasnet-captchas
Всего у нас 271403 семплов. Из особенностей - имеется одна капча в решении, которая выбрасывается автором, предстоит узнать, почему. Все капчи имеют английские буквы и цифры. Данные весят около 3 Gb.

## Моделирование

### Бейзлайн
Базовым подходом для распознавания многобуквенных капч считается архитектура CRNN (Convolutional Recurrent Neural Network), которая сочетает сверточные (CNN) и рекуррентные (RNN) сети и долгое время была стандартом для задач оптического распознавания символов (OCR)

### Основная модель
В решении используется LACC(Label Combination Classifier), модель, которая использует предобученную CNN для извлечения визуальных признаков, особенность данной модели, все символы предсказываются за один forward pass, что быстрее, чем последовательная обработка символом в бэйзлайне.

## Внедрение
Думаю, что данную модель можно использовать для обхода капч на сайтах, используя, например rest API. Или в исследовательских целях, тестировать на реальных капчах, доставая их через rest API, для анализа их эффективности.

## Setup
```
git clone https://github.com/pariramb/CAPTCHA-Recognition-CRNN.git
cd CAPTCHA-Recognition-CRNN
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
pre-commit install
```
## Train
```
dvc pull
python scripts/download_data.py
python scripts/train.py

С конкретной конфигурацией
python scripts/train.py data.batch_size=32 training.epochs=10
```

## Production preparation
python scripts/export_model.py --format onnx

## Infer
python scripts/infer.py --image path/to/image.png

**Формат данных**

Поддерживаемые форматы:
    1. PNG изображения
    2. JPG изображения

**Логирование**
    1. Метрики логируются в MLflow (http://127.0.0.1:8080)
    2. Графики сохраняются в plots/
    3. Конфигурация в configs/