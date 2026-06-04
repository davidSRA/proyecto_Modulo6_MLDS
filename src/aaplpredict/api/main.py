from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aaplpredict.api.predictor import predict
from aaplpredict.api.schemas import PredictionInput, PredictionOutput

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="AAPLPredict API",
    description="Predicción de retornos y volatilidad de AAPL usando XGBoost, MLP o LSTM.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def history(dias: int = 45):
    """Devuelve los últimos N días hábiles de precios de cierre de AAPL."""
    hoy    = date.today()
    inicio = hoy - timedelta(days=dias * 2)   # factor 2 para cubrir fines de semana
    df = yf.download("AAPL", start=str(inicio), end=str(hoy), progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(dias).reset_index(drop=True)
    return {
        "dates":  df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": [round(float(p), 2) for p in df["Close"].tolist()],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(data: PredictionInput) -> PredictionOutput:
    try:
        return predict(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
