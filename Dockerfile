FROM python:3.11-slim

LABEL maintainer="zied-convergen"
LABEL description="Churn Prediction API – FastAPI + MLflow"

WORKDIR /app

COPY requirements-api.txt .
COPY app.py .
COPY model_pipeline.py .
COPY classifier.joblib .

RUN pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
