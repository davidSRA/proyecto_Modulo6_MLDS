"""Reentrenar los mejores modelos y registrarlos en MLflow."""
import runpy

if __name__ == "__main__":
    runpy.run_module("aaplpredict.training.entrenamiento_final", run_name="__main__")
