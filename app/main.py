from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException

from app.schemas import PredictRequest, PredictResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "registry" / "model.joblib"

LABELS = {0: "hockey", 1: "space"}

app = FastAPI(title="Topic Classifier API", version="1.0.0")


@lru_cache
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_model():
    try:
        return _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_model)) -> PredictResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    proba = model.predict_proba([req.text])[0]
    idx = int(proba.argmax())
    return PredictResponse(label=LABELS.get(idx, str(idx)), confidence=float(proba[idx]))
