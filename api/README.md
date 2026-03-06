# 🏥 API de Classificação de Saúde Fetal

## O que é?

Uma aplicação FastAPI para fazer predições de saúde fetal usando um modelo de machine learning treinado com Gradient Boosting. O modelo classifica o estado fetal em 3 categorias: Normal, Suspeito ou Patológico.

## Arquitetura

```
api/
├── main.py                 # Aplicação principal FastAPI
├── __init__.py
└── README.md              # Este arquivo
```

## 🚀 Início Rápido

### 1. Configurar Arquivo `.env`

Na raiz do projeto, crie um arquivo `.env`:

```env
MLFLOW_TRACKING_USERNAME=seu_usuario_dagshub
MLFLOW_TRACKING_PASSWORD=seu_token_dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/seu_usuario_dagshub/seu_repositorio.mlflow
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Iniciar a API

```bash
# Opção 1: Usando o script run_api.py
python run_api.py

# Opção 2: Usando uvicorn diretamente
uvicorn api.main:app --reload
```

A API estará disponível em: **http://localhost:8000**

---

## 📡 API Endpoints

### GET `/health`

Verifica se a API está funcionando e se o modelo foi carregado.

```bash
curl -X GET "http://localhost:8000/health"
```

**Resposta:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### POST `/predict`

Realiza predição de saúde fetal.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [120, 0.5, 0.2, 0.1, 0, 0, 0.1, 0.5, 0.05, 0.3, 30, 100, 160, 5, 3, 130, 140, 135, 50, 0, 0]
  }'
```

**Resposta:**
```json
{
  "prediction": 1,
  "confidence": 0.95,
  "probabilities": {
    "1": 0.95,
    "2": 0.04,
    "3": 0.01
  }
}
```

---

## 📚 Documentação Completa

Para documentação mais detalhada, consulte: [API_USAGE_GUIDE.md](../API_USAGE_GUIDE.md)

## 🎯 Interpretação dos Resultados

- **Classe 1**: Normal 🟢
- **Classe 2**: Suspeito 🟡
- **Classe 3**: Patológico 🔴

A **confiança** varia de 0 a 1, onde 1 é a máxima confiança.

---

## 📖 Documentação Automática

Após iniciar a API, você pode acessar:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔧 Configuração Avançada

### Mudar Porta

```bash
python run_api.py 8888
```

### Com Reload Automático

```bash
uvicorn api.main:app --reload --port 8000
```

### Workers Múltiplos (Produção)

```bash
uvicorn api.main:app --workers 4 --port 8000
```

---

## 🐳 Usando Docker

```bash
# Build
docker build -t fetal-health-api .

# Run
docker run -p 8000:8000 --env-file .env fetal-health-api
```

---

## 📦 Dependências

- **fastapi**: Framework web
- **uvicorn**: ASGI server
- **mlflow**: Model registry
- **scikit-learn**: ML models
- **numpy**: Computação numérica
- **pandas**: Manipulação de dados
- **python-dotenv**: Gerenciamento de variáveis de ambiente

---

## ⚠️ Notas Importantes

1. **Arquivo `.env` NUNCA deve ser commitado** com credenciais reais
2. O modelo é carregado automaticamente no startup da aplicação
3. Todas as 21 features são obrigatórias na predição
4. A resposta é sempre em formato JSON

---

## 🐛 Troubleshooting

| Erro | Solução |
|------|---------|
| Porta 8000 em uso | Use `python run_api.py 8888` |
| Variáveis de ambiente não encontradas | Verifique arquivo `.env` |
| Nenhum modelo encontrado | Verifique credenciais do MLflow |
| Erro ao fazer predição | Verifique se tem exatamente 21 features |

---

## 📞 Contato

- **Autor**: Renan Santos Mendes
- **Email**: renansantosmendes@gmail.com

---

**Versão**: 1.0.0  
**Última atualização**: Março 2026
