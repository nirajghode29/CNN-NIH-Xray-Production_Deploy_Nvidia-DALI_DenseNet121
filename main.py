from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from model import load_trained_model
from labels import LABELS
from schemas import PredictionResponse
from utils import preprocess_image

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    try:
        MODEL = load_trained_model(device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print("Startup model load failed:", repr(e))
        raise
    yield
    MODEL = None


app = FastAPI(
    title="NIH Chest X-ray Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "NIH Chest X-ray API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    x = preprocess_image(contents).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    probabilities = {label: float(prob) for label, prob in zip(LABELS, probs)}
    predicted_labels = [label for label, prob in probabilities.items() if prob >= threshold]

    return PredictionResponse(
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        threshold=threshold,
    )