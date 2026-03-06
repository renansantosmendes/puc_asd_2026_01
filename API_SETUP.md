# API Setup - MLflow/DagsHub

## 1. Configure as Variáveis de Ambiente

Crie um arquivo `.env`:

```bash
DAGSHUB_USERNAME=renansantosmendes
DAGSHUB_PASSWORD=seu_token_aqui
DAGSHUB_REPO=puc_asd_2026_01
```

## 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

## 3. Rode a API

```bash
python run_api.py
```

A API estará disponível em `http://localhost:8000`

## 4. Teste os Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predição
curl -X POST http://localhost:8000/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [120.0, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0, 0.0]}'
```

## 5. Documentação Interativa

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

- `GET /` - Root
- `GET /health` - Health check
- `GET /model-info` - Info do modelo
- `POST /predictions/predict` - Predição única
- `POST /predictions/predict-batch` - Predição em batch
