# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** AAPLPredict v0.1.0 – Predicción de retornos y volatilidad de AAPL
- **Plataforma de despliegue:** Railway Cloud Platform
- **Requisitos técnicos:**
- **Software**:
- Python
- Fast API
- Pandas Numpy
- Scikit-Learn
- XGBoost
- TensorFlow
- Optuna
- MLflow
- yfinance
- **Harduware minimo**:
- CPU
- Memoria Ram: 2 GB
- Disco duro: 5 GB
- GPU: No requerida.
- HTTP.

- **Requisitos de seguridad:**
- Conunicacion mediante HTTPS.
- Validación de datos mediante Pydantic
- Restriccion de archivos de acceso mediante static,
- No se almacenan credenciales de usuario.
- Protecciones de solicitudes mal formadas mediante validaciones FastAPI

- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

## Código de despliegue

- **Archivo principal:** aaplpredict/api/main.py
- **Rutas de acceso a los archivos:** (lista de rutas de acceso a los archivos necesarios para el despliegue)
- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)

## Documentación del despliegue

- **Instrucciones de instalación:** (instrucciones detalladas para instalar el modelo en la plataforma de despliegue)
- **Instrucciones de configuración:** (instrucciones detalladas para configurar el modelo en la plataforma de despliegue)
- **Instrucciones de uso:** (instrucciones detalladas para utilizar el modelo en la plataforma de despliegue)
- **Instrucciones de mantenimiento:** (instrucciones detalladas para mantener el modelo en la plataforma de despliegue)
