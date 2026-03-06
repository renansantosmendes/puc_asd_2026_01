# FastAPI - Fetal Health Prediction API

## 🚀 Quick Start

### Instalação
```bash
pip install -r requirements.txt
```

### Rodar a API

#### Opção 1: Localmente
```bash
python run_api.py
```

Acesse em: `http://localhost:8000`

#### Opção 2: Docker
```bash
docker-compose up
```

#### Opção 3: Porta customizada
```bash
python run_api.py 8080
```

## 📚 Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Informações do Modelo
```bash
GET /model-info
```
Response:
```json
{
  "model_type": "GradientBoostingClassifier",
  "features_count": 21,
  "classes": ["Normal", "Suspeito", "Patológico"]
}
```

### Predição Única
```bash
POST /predictions/predict
```
Request:
```json
{
  "features": [120.0, 0.5, 0.3, 0.2, 0.1, 0.08, ...]
}
```
Response:
```json
{
  "prediction": 1,
  "confidence": 0.95,
  "probabilities": {
    "1_normal": 0.95,
    "2_suspeito": 0.04,
    "3_patologico": 0.01
  }
}
```

### Predição em Batch
```bash
POST /predictions/predict-batch
```
Request:
```json
{
  "samples": [
    [120.0, 0.5, 0.3, ...],
    [125.0, 0.6, 0.35, ...]
  ]
}
```
Response:
```json
{
  "predictions": [1, 2, 1],
  "confidences": [0.95, 0.85, 0.92],
  "total_samples": 3
}
```

## 🧪 Testar API

```bash
python test_api_client.py
```

## 📖 Documentação Interativa

Swagger UI: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

## 🐳 Docker

### Build
```bash
docker build -t fetal-health-api .
```

### Run
```bash
docker run -p 8000:8000 fetal-health-api
```

### Compose
```bash
docker-compose up
docker-compose down
```

## 📋 Estrutura da API

```
api/
├── main.py              # Aplicação principal
├── schemas/
│   └── prediction.py    # Pydantic models
└── routes/
    ├── predictions.py   # Rotas de predição
    └── health.py        # Rotas de saúde
```

## ⚙️ Configurações

- **Host**: 0.0.0.0 (aceita conexões de qualquer lugar)
- **Porta**: 8000 (default)
- **Reload**: Habilitado (desenvolvimento)
- **CORS**: Habilitado (aceita requisições de qualquer origem)

## 🔒 Segurança em Produção

Para produção, ajuste em `api/main.py`:

```python
# CORS - Restringir origens
allow_origins=["https://seu-dominio.com"]

# Reload - Desabilitar
reload=False

# Log level
log_level="warning"
```

## 📊 Performance

- **Predição Única**: ~10-50ms
- **Batch (100 amostras)**: ~50-100ms
- **Batch (1000 amostras)**: ~300-500ms

## 🐛 Troubleshooting

### Erro: Modelo não carregado
```
Certifique-se de que 'asd_model.pkl' está no diretório raiz
```

### Erro: Conexão recusada
```
Verifique se a API está rodando na porta correta
```

### Erro: 21 features esperadas
```
Verifique se todas as 21 features estão sendo enviadas
```

## 📝 Exemplo com cURL

```bash
# Predição única
curl -X POST "http://localhost:8000/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [120.0, 0.5, 0.3, ..., 0.0, 0.0]}'

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info
```

## 📝 Exemplo com Python

```python
import requests

url = "http://localhost:8000/predictions/predict"
data = {
    "features": [120.0, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01,
                 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0, 0.0]
}
response = requests.post(url, json=data)
print(response.json())
```
