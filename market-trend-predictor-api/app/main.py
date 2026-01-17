from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from app.utils import load_model
from app.ml import fetch_data, make_features, create_trend_labels, predict_next_day

app = FastAPI(
    title="Market Trend Predictor API",
    description="AI Tool: Predict next trading day's market trend regime (Uptrend/Downtrend/Sideways).",
    version="1.0"
)

MODEL_PATH = "model/rf_nextday.pkl"
model = load_model(MODEL_PATH)

class PredictRequest(BaseModel):
    ticker: str = Field(default="^NSEI")
    start: str = Field(default="2015-01-01")
    end: str = Field(default="2024-01-01")
    threshold: float = Field(default=0.002)

class PredictResponse(BaseModel):
    ticker: str
    last_available_date: str
    predicted_for: str
    trend_class: int
    trend_label: str
    confidence: Optional[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-next", response_model=PredictResponse)
def api_predict_next(req: PredictRequest):
    raw = fetch_data(req.ticker, req.start, req.end)
    feat = make_features(raw)
    labeled = create_trend_labels(feat, threshold=req.threshold)

    out = predict_next_day(model, labeled)

    return PredictResponse(
        ticker=req.ticker,
        last_available_date=out["last_available_date"],
        predicted_for=out["predicted_for"],
        trend_class=out["trend_class"],
        trend_label=out["trend_label"],
        confidence=out["confidence"]
    )
@app.get("/")
def root():
    return {
        "message": "Market Trend Predictor API",
        "docs": "/docs"
    }
