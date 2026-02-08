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

### Install Dependencies

The project uses `uv` for dependency management. Install it and sync the environment:

```bash
# Install uv (if not already installed)
pip install uv

# Navigate to the project directory and install dependencies
cd /path/to/project
uv sync
```

### Download Dataset

```bash
cd /path/to/project
export KAGGLE_API_TOKEN=YOUR_TOKEN
uv run -m captcha_rec.commands download_data data.dvc_storage=/full/path/to/dataset
```

**Notes:**

- data.dvc_storage - path to the directory where the dvc storage will be stored

- Data is automatically downloaded via DVC when needed

- A configured Kaggle API token is required

## Train

### Start MLflow for Experiment Tracking

MLflow is used for logging metrics, parameters, and artifacts:

```bash
cd /path/to/project
uv run mlflow server --host 127.0.0.1 --port 8080
```

### Start Training

Run model training with specified parameters:

```bash
cd /path/to/project
export KAGGLE_API_TOKEN=YOUR_TOKEN
uv run -m captcha_rec.commands train \
  trainer.max_epochs=6 \
  model.lr=0.0002 \
  data.dvc_storage=/path/to/dataset
```

**Main parameters:**

- `trainer.max_epochs` - number of training epochs

- `model.lr` - learning rate

- `data.dvc_storage` - path to the dataset

**What happens during training:**

1. Automatic dataset download if needed

2. DataModule initialization for data processing

3. LACCModule model creation

4. MLflow logging setup

5. Callbacks used:
   - ModelCheckpoint for saving best weights

   - LearningRateMonitor for tracking learning rate changes

6. Training plots saved to the plots/ directory

## Production preparation

### Export to ONNX

Export the trained model to ONNX format for optimized inference:

```bash
uv run -m captcha_rec.commands export_onnx
```

**Functionality:**

- Automatically finds the latest checkpoint in checkpoints/

- Exports the model with correct input/output dimensions

- Saves the model to the path specified in the configuration

### Export to TensorRT

Convert ONNX model to TensorRT engine for maximum performance:

```bash
uv run -m captcha_rec.commands export_trt
```

**Configuration parameters:**

- `export.trt_fp16` - use half precision

- `export.trt_workspace_mb` - workspace memory size

### Register Model in MLflow Model Registry

Prepare the model for serving via MLflow:

```bash
uv run -m captcha_rec.commands register_model_mlflow
```

### MLflow Serving

Start a REST API server for model serving:

```bash
uv run -m captcha_rec.infer.mlflow_api
```

### Test MLflow Serving

Test script to verify serving functionality:

```bash
uv run -m captcha_rec.infer.mlflow_test
```

## Inference

### Batch Inference

Run predictions on multiple images:

```
uv run -m captcha_rec.commands infer infer.output=outputs/preds.jsonl
```

**Configuration parameters:**

- `infer.inputs` - list of paths to input images

- `infer.output` - path to output JSONL file

- `infer.onnx_path` - path to ONNX model

**Output data format (JSONL):**

```
{"image_path": "path/to/image.png", "prediction": "ABCD", "confidence": 0.95}
```

## Batch Inference

All commands are implemented in the Commands class with the following methods:

1. `download_data` - download dataset

2. `train` - complete training cycle with logging

3. `export_onnx` - export model to ONNX format

4. `export_trt` - convert to TensorRT engine

5. `infer` - batch inference on images

6. `register_model_mlflow` - register model in MLflow

## Configuration Management

The project uses Hydra for configuration management. Main configs are located in `configs/`

- `train.yaml` - training configuration

- `infer.yaml` - inference configuration

Parameters can be overridden via command line, for example:

```bash
uv run -m captcha_rec.commands infer infer.output=outputs/preds.jsonl
```

## Training Visualization

After training, plots are automatically generated in the `plots/` directory:

- `loss_curves.png` - train/validation loss curves

- `val_char_acc.png` - character recognition accuracy

- `val_seq_acc.png` - sequence recognition accuracy
