import numpy as np
import pandas as pd
import yfinance as yf

TREND_MAP = {1: "Uptrend", 0: "Sideways", -1: "Downtrend"}

FEATURES = [
    "Return", "LogReturn",
    "Volatility_20", "Volatility_50",
    "Momentum_10", "ROC_10",
    "Vol_Change", "Vol_MA_20",
    "Volume"
]

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df is None or len(df) == 0:
        raise ValueError("No data returned from yfinance.")
    return df

def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Volatility_20"] = df["Return"].rolling(20).std()
    df["Volatility_50"] = df["Return"].rolling(50).std()

    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["ROC_10"] = df["Close"].pct_change(10)

    df["Vol_Change"] = df["Volume"].pct_change()
    df["Vol_MA_20"] = df["Volume"].rolling(20).mean()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

def create_trend_labels(df_feat: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
    df = df_feat.copy()
    diff = (df["MA_20"] - df["MA_50"]) / df["MA_50"]

    df["Trend"] = np.where(
        diff > threshold, 1,
        np.where(diff < -threshold, -1, 0)
    )
    df["Trend_Label"] = df["Trend"].map(TREND_MAP)

    return df

def predict_next_day(model, df_labeled: pd.DataFrame):
    last_date = str(df_labeled.index[-1].date())

    X_last = df_labeled[FEATURES].iloc[[-1]]
    pred_class = int(model.predict(X_last)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_last)[0]
        class_to_prob = dict(zip(model.classes_, proba))
        confidence = float(class_to_prob.get(pred_class, 0.0))

    return {
        "last_available_date": last_date,
        "predicted_for": "next_trading_day",
        "trend_class": pred_class,
        "trend_label": TREND_MAP[pred_class],
        "confidence": confidence
    }
