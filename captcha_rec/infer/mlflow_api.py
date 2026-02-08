import io
import logging

import mlflow.pyfunc
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from captcha_rec.data.datamodule import build_default_vocab

app = FastAPI(title="CAPTCHA Recognition API")

model = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(model_uri="models:/captcha_crnn/1")
        logging.info("Model loaded successfully from MLflow")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        model = None


@app.get("/health")
async def health_check():
    return {"status": "healthy" if model else "unhealthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image = image.resize((256, 256))

        image_array = np.array(image)

        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        image_array = image_array.transpose(2, 0, 1) / 255.0

        image_array = np.expand_dims(image_array, 0)

        predictions = model.predict(image_array)

        print(predictions["logits"][0].shape)
        predicted_text = decode_predictions(predictions["logits"][0])

        return {
            "filename": file.filename,
            "prediction": predicted_text,
            "confidence": float(np.max(predictions["logits"][0])),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def decode_predictions(prediction: np.ndarray) -> str:
    char_indices = np.argmax(prediction, axis=0)
    vocab = build_default_vocab()

    result = []
    for idx in char_indices:
        if idx < vocab.size():
            result.append(vocab.tokens[idx])

    return "".join(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
