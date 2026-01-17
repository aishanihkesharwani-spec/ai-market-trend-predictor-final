import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


# =========================================================
# HELPERS
# =========================================================
FASTAPI_URL = "http://127.0.0.1:8000"

TREND_LABEL = {1: "Uptrend", 0: "Sideways", -1: "Downtrend"}


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix yfinance returning MultiIndex columns sometimes."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def trend_to_color(t: int) -> str:
    """Map trend class to colors."""
    if t == 1:
        return "rgba(0,180,0,0.9)"     # green
    if t == -1:
        return "rgba(255,0,0,0.85)"    # red
    return "rgba(255,165,0,0.85)"      # orange


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger(close: pd.Series, window: int = 20):
    """Bollinger Bands."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return ma, upper, lower


def build_rule_trend(df: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
    """
    Rule trend using MA20 vs MA50.
    Output Trend in {-1,0,1}.
    """
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out = out.dropna()

    diff = (out["MA20"] - out["MA50"]) / out["MA50"]
    out["Trend"] = 0
    out.loc[diff > threshold, "Trend"] = 1
    out.loc[diff < -threshold, "Trend"] = -1

    return out


def show_trend_card(trend_label: str, confidence):
    if trend_label.lower() == "uptrend":
        st.success("📈 UP TREND")
    elif trend_label.lower() == "downtrend":
        st.error("📉 DOWN TREND")
    else:
        st.warning("➡️ SIDEWAYS / RANGE")

    if confidence is not None:
        conf = float(confidence)
        st.metric("Confidence", f"{conf * 100:.2f}%")
        st.progress(min(max(conf, 0.0), 1.0))


# =========================================================
# PAGE CONFIG + HEADER
# =========================================================
st.set_page_config(page_title="AI Market Trend Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <div style="padding:18px;border-radius:18px;background:linear-gradient(90deg,#111827,#1f2937,#0f172a);color:white;">
        <h1 style="margin:0;text-align:center;">📈 AI Market Trend Predictor</h1>
        <p style="margin:0;text-align:center;font-size:17px;opacity:0.9;">
        Predict the <b>next trading day trend</b> (Uptrend / Downtrend / Sideways) for NIFTY / stocks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input("Ticker", value="^NSEI")
start = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")
end = st.sidebar.text_input("End date (YYYY-MM-DD)", value="2024-01-01")

threshold = st.sidebar.slider(
    "Sideways threshold (MA20 vs MA50)",
    0.0005, 0.01, 0.002, 0.0005
)

timeline_days = st.sidebar.selectbox(
    "Timeline window (days)",
    [30, 60, 120, 180, 250],
    index=3
)

predict_btn = st.sidebar.button("🚀 Predict Next Day")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Educational prototype. Not financial advice.")

# =========================================================
# MAIN LAYOUT
# =========================================================
col1, col2 = st.columns([1, 1.7], gap="large")

# =========================================================
# LEFT: AI PREDICTION PANEL (+ DOWNLOAD REPORT)
# =========================================================
with col1:
    st.subheader("🔮 AI Prediction")
    st.info("Click **Predict Next Day** to get the model forecast.")

    if predict_btn:
        payload = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "threshold": float(threshold),
        }

        try:
            with st.spinner("Calling AI model..."):
                r = requests.post(f"{FASTAPI_URL}/predict-next", json=payload, timeout=60)

            if r.status_code != 200:
                st.error(f"Backend error {r.status_code}: {r.text}")
            else:
                out = r.json()

                # Save to session for Model vs Rule section
                st.session_state["model_trend_label"] = out.get("trend_label", None)
                st.session_state["model_confidence"] = out.get("confidence", None)

                st.markdown("### ✅ Result")
                show_trend_card(out["trend_label"], out.get("confidence", None))

                st.markdown("### 📌 Details")
                st.write(
                    {
                        "Ticker": out["ticker"],
                        "Last available date": out["last_available_date"],
                        "Predicted for": out["predicted_for"],
                        "Trend class": out["trend_class"],
                    }
                )

                # ✅ Download Report (Feature #4)
                report = {
                    "ticker": out["ticker"],
                    "last_available_date": out["last_available_date"],
                    "predicted_for": out["predicted_for"],
                    "trend_label": out["trend_label"],
                    "trend_class": out["trend_class"],
                    "confidence": out.get("confidence", None),
                    "threshold": float(threshold),
                    "start": start,
                    "end": end
                }
                report_df = pd.DataFrame([report])

                st.download_button(
                    "⬇️ Download Prediction Report (CSV)",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_trend_prediction.csv",
                    mime="text/csv"
                )

        except requests.exceptions.ConnectionError:
            st.error("FastAPI backend not running.\n\nRun: `py -m uvicorn app.main:app --reload`")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# =========================================================
# RIGHT: CHARTS + TECH INDICATORS + HEAT STRIP + COMPARISON
# =========================================================
with col2:
    st.subheader("📊 Visual Dashboard")

    try:
        # ---------- Download ONCE ----------
        with st.spinner("Downloading market data..."):
            df = yf.download(ticker, start=start, end=end)

        df = flatten_yf_columns(df)

        if df is None or df.empty or "Close" not in df.columns:
            st.warning("No chart data returned. Check ticker/dates.")
        else:
            df = df.copy()
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna()

            # =========================================================
            # PRICE CHART (Close + MA20/MA50)
            # =========================================================
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
            fig_price.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
            fig_price.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))

            fig_price.update_layout(
                height=420,
                template="plotly_white",
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Date",
                yaxis_title="Price",
                title="Price + Moving Averages"
            )

            st.plotly_chart(fig_price, use_container_width=True)

            # =========================================================
            # RULE TREND + TIMELINE GRAPH
            # =========================================================
            df_tr = build_rule_trend(df, threshold=float(threshold))
            df_last = df_tr.tail(int(timeline_days))

            st.markdown("### 📈 Trend Timeline (Rule-based)")

            if df_last.empty:
                st.warning("Not enough data to display trend timeline.")
            else:
                fig_trend = go.Figure()
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_last.index,
                        y=df_last["Trend"],
                        mode="lines+markers",
                        line=dict(width=4),
                        marker=dict(size=7),
                        name="Trend",
                    )
                )

                fig_trend.update_layout(
                    height=260,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Date",
                    yaxis_title="Trend",
                )

                fig_trend.update_yaxes(
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=["Downtrend", "Sideways", "Uptrend"],
                    range=[-1.2, 1.2],
                )

                st.plotly_chart(fig_trend, use_container_width=True)

                # =========================================================
                # FEATURE #1: TREND HEAT STRIP
                # =========================================================
                st.markdown("### 🟩🟧🟥 Trend Regime Strip")

                strip = df_last[["Trend"]].copy()
                strip["color"] = strip["Trend"].apply(trend_to_color)

                fig_strip = go.Figure()
                fig_strip.add_trace(
                    go.Bar(
                        x=strip.index,
                        y=[1] * len(strip),
                        marker_color=strip["color"],
                        hovertext=strip["Trend"].map({1: "Uptrend", 0: "Sideways", -1: "Downtrend"}),
                        hoverinfo="text+x",
                        name="TrendStrip",
                    )
                )

                fig_strip.update_layout(
                    height=160,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(visible=False),
                    xaxis_title="Date",
                    showlegend=False,
                )
                st.plotly_chart(fig_strip, use_container_width=True)

                # =========================================================
                # FEATURE #2: MODEL vs RULE COMPARISON
                # =========================================================
                st.markdown("### 🤖 Model vs Rule Comparison")

                rule_latest = int(df_last["Trend"].iloc[-1])
                rule_label = TREND_LABEL[rule_latest]

                model_label = st.session_state.get("model_trend_label", None)
                model_conf = st.session_state.get("model_confidence", None)

                cA, cB = st.columns(2)
                cA.metric("Rule Latest Trend", rule_label)
                if model_label:
                    cB.metric("Model Predicts Next Day", model_label)
                else:
                    cB.metric("Model Predicts Next Day", "Run prediction ⬅️")

                if model_conf is not None:
                    st.caption("Model Confidence")
                    st.progress(min(max(float(model_conf), 0.0), 1.0))

            # =========================================================
            # FEATURE #5: TECHNICAL INDICATOR PANEL
            # =========================================================
            st.markdown("### 🧠 Technical Indicator Panel (RSI + Bollinger)")

            df_ind = df.copy()
            df_ind["RSI14"] = compute_rsi(df_ind["Close"], period=14)
            df_ind["BB_MA20"], df_ind["BB_Upper"], df_ind["BB_Lower"] = compute_bollinger(df_ind["Close"], window=20)
            df_ind = df_ind.dropna()

            if df_ind.empty:
                st.warning("Not enough data for indicators.")
            else:
                # Mini metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Latest Close", f"{df_ind['Close'].iloc[-1]:.2f}")
                m2.metric("RSI(14)", f"{df_ind['RSI14'].iloc[-1]:.2f}")
                bb_width = (df_ind["BB_Upper"].iloc[-1] - df_ind["BB_Lower"].iloc[-1]) / df_ind["BB_MA20"].iloc[-1]
                m3.metric("BB Width", f"{bb_width*100:.2f}%")

                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI14"], name="RSI(14)"))
                fig_rsi.update_layout(
                    height=260,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Date",
                    yaxis_title="RSI",
                )
                fig_rsi.add_hline(y=70)
                fig_rsi.add_hline(y=30)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # Bollinger Bands Chart
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"], name="Close"))
                fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_MA20"], name="BB MA20"))
                fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="Upper Band"))
                fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="Lower Band"))

                fig_bb.update_layout(
                    height=320,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Date",
                    yaxis_title="Price",
                )
                st.plotly_chart(fig_bb, use_container_width=True)

    except Exception as e:
        st.error(f"Dashboard error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using FastAPI + Streamlit • Educational Prototype")

