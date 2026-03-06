from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
scaler = None


@app.on_event("startup")
async def startup():
    global model, scaler
    
    # Carregar variáveis de ambiente
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    
    if not all([mlflow_username, mlflow_password, mlflow_uri]):
        print("Erro: Variáveis de ambiente do MLflow não configuradas no arquivo .env")
        return
    
    # Configurar variáveis de ambiente
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"✓ MLflow configurado para: {mlflow_uri}")
    
    client = MlflowClient()
    logged_models = client.search_logged_models(['0'])
    
    if logged_models:
        model = mlflow.pyfunc.load_model(logged_models[0].model_uri)
        print(f"✓ Modelo carregado de: {logged_models[0].model_uri}")
    else:
        print("⚠ Nenhum modelo encontrado no MLflow")
    
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame([[0]*21]))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    return {
        "prediction": int(pred),
        "confidence": float(np.max(proba)),
        "probabilities": {"1": float(proba[0]), "2": float(proba[1]), "3": float(proba[2])}
    }
