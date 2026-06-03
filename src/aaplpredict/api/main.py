from fastapi import FastAPI, HTTPException

from aaplpredict.api.predictor import predict
from aaplpredict.api.schemas import PredictionInput, PredictionOutput

app = FastAPI(
    title="AAPLPredict API",
    description="Predicción de retornos y volatilidad de AAPL usando XGBoost, MLP o LSTM.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(data: PredictionInput) -> PredictionOutput:
    try:
        return predict(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
