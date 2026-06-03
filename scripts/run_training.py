"""Ejecuta la búsqueda de hiperparámetros con Optuna."""
from aaplpredict.training import main  # noqa: F401 — ejecuta el bloque __main__

import runpy
import os

if __name__ == "__main__":
    runpy.run_module("aaplpredict.training.main", run_name="__main__")
