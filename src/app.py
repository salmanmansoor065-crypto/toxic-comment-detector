"""
FastAPI app for Toxic Comment Detection
Endpoints: / (home), /predict, /health
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Config ----
MODEL_PATH = "/app/models/v2_bilstm.keras"
TOKENIZER_PATH = "/app/models/tokenizer_v2.pkl"
MAX_LEN = 120
THRESHOLD = 0.3
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---- Load model + tokenizer ----
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

app = FastAPI(title="Toxic Comment Detector")

class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    comment: str
    predictions: dict
    is_toxic: bool

@app.get("/")
def home():
    return {
        "message": "Toxic Comment Detection API",
        "usage": "POST /predict with {'text': 'your comment here'}",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "v2-bidirectional-lstm"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CommentRequest):
    # Preprocess
    seq = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    # Predict
    proba = model.predict(padded, verbose=0)[0]
    predictions = {label: round(float(prob), 4) for label, prob in zip(LABELS, proba)}
    is_toxic = any(prob > THRESHOLD for prob in proba)
    
    return PredictionResponse(
        comment=request.text,
        predictions=predictions,
        is_toxic=is_toxic
    )