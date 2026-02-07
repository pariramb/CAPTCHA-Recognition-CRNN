# CAPTCHA-Recognition-CRNN

## Problem Statement

Recognizing captchas from images. Multi-letter captchas are considered. A solution from Kaggle is used. https://www.kaggle.com/code/lapl04/pytorch-captcha-recognizer/notebook#Save

### Input and Output Data Format

The input is a photo of the captcha, the output is the text version of this captcha

### Metrics

This solution uses 5 different datasets. 90% is used for training. 10% is used for validation. At the end, there is a test on the validation data, which needs to be modified. To ensure reproducibility, we fix the seed. Reproducibility is important for analyzing the resulting solution and debugging.

### Validation

We will use 5 different datasets. 90% is used for training. 10% is used for validation. Additionally, the proposed solution includes a test on the validation data at the end, which needs to be modified. I suggest using 80% for training, 10% for validation, and 10% for testing. To ensure reproducibility, we will fix the seed. Reproducibility is crucial for analyzing the solution and enabling debugging.

### Data

References:
https://www.kaggle.com/datasets/akashguna/large-captcha-dataset
www.kaggle.com/datasets/parsasam/captcha-dataset
https://www.kaggle.com/datasets/aadhavvignesh/captcha-images
https://www.kaggle.com/datasets/fournierp/captcha-version-2-images
https://www.kaggle.com/datasets/jassoncarvalho/comprasnet-captchas
In total we have 271403 samples. Of the features - there is one captcha in the solution, which is thrown out by the author, it is to be found out why. All captchas have English letters and numbers. The data weighs about 3 Gb.

## Modeling

### Baseline

The basic approach for recognizing multi-letter captchas is the CRNN (Convolutional Recurrent Neural Network) architecture, which combines convolutional (CNN) and recurrent (RNN) networks and has long been the standard for optical character recognition (OCR) tasks.

### Main model

The solution uses LACC (Label Combination Classifier), a model that uses a pre-trained CNN to extract visual features. This model predicts all characters in a single forward pass, which is faster than processing characters sequentially in the baseline.

### Deployment

I think that this model can be used to bypass captchas on websites using, for example, the rest API. Or, for research purposes, to test it on real captchas by retrieving them through the rest API to analyze their effectiveness.

## Setup

pip install uv

```bash
cd /path/to/project
uv sync
```

## Train
Datasets take from Kaggle, you need to take a token
Необходимо заполнить svhn.yaml
Запустить `mlflow`
```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

```bash
export KAGGLE_API_TOKEN=YOUR_TOKEN
uv run -m captcha_rec.commands train trainer.max_epochs=6 model.lr=0.0002
```

Сперва будет скачан датасет в x
## Production preparation

## Infer

# captcha-rec

Проект: распознавание символов на изображениях (OCR) на основе нейросети **LACC**:

- backbone: `efficientnet_v2_m().features`
- преобразование признаков через матрицу `converter` (как в исходном коде)
- предсказание последовательности токенов длины `max_len` (с `<pad>`)

Реализовано:

- обучение: PyTorch Lightning
- конфиги: Hydra (`configs/`)
- данные: DVC (встроено в команды train/infer) или bootstrap через `download_data()`
- логирование: MLflow (метрики + гиперпараметры + git commit id)
- production: export ONNX + export TensorRT (через `trtexec`)
- inference: onnxruntime (отдельный лёгкий код)
- inference server: подготовка репозитория Triton (`prepare_triton_repo`)
- качество кода: pre-commit (black/isort/flake8/prettier)

---

## Setup

```bash
poetry install
poetry run pre-commit install

# MLflow (локально):
poetry run mlflow server --host 127.0.0.1 --port 8080

# DVC:
poetry run dvc init
# настроить remote (gdrive/s3/local) и сделать dvc add/push данных и артефактов
```

    """
    Единая точка входа.
    Пример:
      python -m captcha_rec.commands train trainer.max_epochs=2 model.lr=0.0002
      python -m captcha_rec.commands export_onnx export.onnx_path=artifacts/model.onnx
      python -m captcha_rec.commands infer infer.inputs=[data/examples] infer.output=outputs/preds.jsonl
    """
