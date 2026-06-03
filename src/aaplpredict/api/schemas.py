from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    modelo: str = Field(
        default="XGBoost",
        description="Modelo a usar para la predicción: 'XGBoost', 'MLP' o 'LSTM'.",
        examples=["XGBoost", "MLP", "LSTM"],
    )


class PredictionOutput(BaseModel):
    modelo_usado:        str   = Field(description="Nombre del modelo que generó la predicción")
    precio_actual:       float = Field(description="Último precio de cierre conocido (USD)")
    precio_1d:           float = Field(description="Precio estimado en 1 día (USD)")
    precio_5d:           float = Field(description="Precio estimado en 5 días (USD)")
    intervalo_5d_sup:    float = Field(description="Límite superior ±1σ del precio a 5 días (USD)")
    intervalo_5d_inf:    float = Field(description="Límite inferior ±1σ del precio a 5 días (USD)")
    volatilidad_5d:      float = Field(description="Volatilidad realizada esperada (std log-returns 5d)")
