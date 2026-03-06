# 🏥 Guia de Uso da API de Classificação de Saúde Fetal

## 📋 Índice

1. [Configuração Inicial](#configuração-inicial)
2. [Iniciando a API](#iniciando-a-api)
3. [Endpoints Disponíveis](#endpoints-disponíveis)
4. [Exemplos de Uso](#exemplos-de-uso)
5. [Interpretação dos Resultados](#interpretação-dos-resultados)
6. [Solução de Problemas](#solução-de-problemas)

---

## 🔧 Configuração Inicial

### 1. Arquivo `.env`

Antes de iniciar a API, certifique-se de que o arquivo `.env` está configurado com suas credenciais do DagsHub/MLflow:

```env
MLFLOW_TRACKING_USERNAME=seu_usuario_dagshub
MLFLOW_TRACKING_PASSWORD=seu_token_dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/seu_usuario_dagshub/seu_repositorio.mlflow
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Iniciando a API

### Opção 1: Usando o script `run_api.py`

```bash
python run_api.py
```

### Opção 2: Usando uvicorn diretamente

```bash
uvicorn api.main:app --reload
```

### Opção 3: Especificar porta customizada

```bash
python run_api.py 8888
```

A API estará disponível em `http://localhost:8000` (ou a porta especificada).

---

## 📡 Endpoints Disponíveis

### 1. Health Check - `GET /health`

Verifica se a API está funcionando e se o modelo foi carregado com sucesso.

**URL:**
```
GET http://localhost:8000/health
```

**Resposta (Sucesso):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Resposta (Modelo não carregado):**
```json
{
  "status": "ok",
  "model_loaded": false
}
```

---

### 2. Predição - `POST /predict`

Realiza a classificação de saúde fetal baseado em características de cardiotocografia.

**URL:**
```
POST http://localhost:8000/predict
```

**Content-Type:**
```
application/json
```

**Corpo da Requisição:**

O corpo deve conter uma lista de 21 valores float representando as características de CTG:

```json
{
  "features": [
    120.0,    // 1. baseline
    0.5,      // 2. acceleration
    0.2,      // 3. fetal_movement
    0.1,      // 4. uterine_contractions
    0.0,      // 5. severe_decelerations
    0.0,      // 6. prolonged_decelerations
    0.1,      // 7. abnormal_short_term_variability
    0.5,      // 8. mean_value_of_short_term_variability
    0.05,     // 9. percentage_of_time_with_abnormal_long_term_variability
    0.3,      // 10. mean_value_of_long_term_variability
    30.0,     // 11. histogram_width
    100.0,    // 12. histogram_min
    160.0,    // 13. histogram_max
    5.0,      // 14. histogram_number_of_peaks
    3.0,      // 15. histogram_number_of_zeroes
    130.0,    // 16. histogram_mode
    140.0,    // 17. histogram_mean
    135.0,    // 18. histogram_median
    50.0,     // 19. histogram_variance
    0.0,      // 20. histogram_tendency
    0.0       // 21. fetal_health (não usado, deixar como 0)
  ]
}
```

**Resposta (Sucesso):**
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

## 💡 Exemplos de Uso

### Usando cURL

#### Health Check:
```bash
curl -X GET "http://localhost:8000/health"
```

#### Predição:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [120, 0.5, 0.2, 0.1, 0, 0, 0.1, 0.5, 0.05, 0.3, 30, 100, 160, 5, 3, 130, 140, 135, 50, 0, 0]
  }'
```

---

### Usando Python (requests)

```python
import requests
import json

# Configuração
BASE_URL = "http://localhost:8000"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print("Health Check:", response.json())

# 2. Fazer Predição
features = [
    120.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.1, 0.5, 0.05, 0.3,
    30.0, 100.0, 160.0, 5.0, 3.0, 130.0, 140.0, 135.0, 50.0, 0.0, 0.0
]

payload = {"features": features}
response = requests.post(f"{BASE_URL}/predict", json=payload)

print("Predição:", json.dumps(response.json(), indent=2))
```

---

### Usando JavaScript/Fetch API

```javascript
const BASE_URL = "http://localhost:8000";

// 1. Health Check
fetch(`${BASE_URL}/health`)
  .then(response => response.json())
  .then(data => console.log("Health:", data));

// 2. Fazer Predição
const features = [
  120, 0.5, 0.2, 0.1, 0, 0, 0.1, 0.5, 0.05, 0.3,
  30, 100, 160, 5, 3, 130, 140, 135, 50, 0, 0
];

fetch(`${BASE_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ features: features })
})
.then(response => response.json())
.then(data => console.log("Predição:", data));
```

---

### Usando Postman

1. **Criar nova request GET:**
   - URL: `http://localhost:8000/health`
   - Enviar

2. **Criar nova request POST:**
   - URL: `http://localhost:8000/predict`
   - Método: POST
   - Headers: `Content-Type: application/json`
   - Body (raw JSON):
   ```json
   {
     "features": [120, 0.5, 0.2, 0.1, 0, 0, 0.1, 0.5, 0.05, 0.3, 30, 100, 160, 5, 3, 130, 140, 135, 50, 0, 0]
   }
   ```

---

## 🔍 Interpretação dos Resultados

### Classificação das Clases

| Classe | Interpretação | Recomendação |
|--------|---------------|--------------|
| **1** | 🟢 **Normal** | Acompanhamento regular |
| **2** | 🟡 **Suspeito** | Acompanhamento mais próximo |
| **3** | 🔴 **Patológico** | Intervenção imediata |

### Confiança (Confidence)

- **0.90 - 1.00**: Muito alta, predição muito segura ✅
- **0.70 - 0.90**: Alta, predição segura ✅
- **0.50 - 0.70**: Moderada, predição razoável ⚠️
- **< 0.50**: Baixa, predição pouco confiável ❌

### Probabilidades

As probabilidades representam a chance do feto estar em cada estado:

```json
{
  "probabilities": {
    "1": 0.85,  // 85% de chance de estar Normal
    "2": 0.10,  // 10% de chance de estar Suspeito
    "3": 0.05   // 5% de chance de estar Patológico
  }
}
```

---

## 🚨 Solução de Problemas

### Problema: "Erro: Variáveis de ambiente do MLflow não configuradas"

**Solução:**
1. Verifique se o arquivo `.env` existe na raiz do projeto
2. Certifique-se de que tem as 3 variáveis configuradas:
   - `MLFLOW_TRACKING_USERNAME`
   - `MLFLOW_TRACKING_PASSWORD`
   - `MLFLOW_TRACKING_URI`

### Problema: "⚠️ Aviso: Nenhum modelo encontrado no MLflow"

**Solução:**
1. Verifique se as credenciais estão corretas no arquivo `.env`
2. Verifique se o modelo foi registrado no MLflow (experimento 0)
3. Certifique-se de que tem acesso à internet e ao DagsHub

### Problema: "Erro ao fazer predição"

**Solução:**
1. Verifique se enviou exatamente 21 características
2. Certifique-se de que todos os valores são números float
3. Verifique se o formato JSON está correto

### Problema: "Porta 8000 já está em uso"

**Solução:**
Use uma porta diferente:
```bash
python run_api.py 8888
```

---

## 📚 Documentação Interativa

A API fornece documentação interativa automática:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Abra qualquer um desses links em seu navegador para ver a documentação interativa completa.

---

## 🔐 Segurança

⚠️ **IMPORTANTE:**

- **Nunca** commite o arquivo `.env` com credenciais reais no Git
- Use o arquivo `.env.example` como template
- Armazene credenciais em variáveis de ambiente em produção
- Considere adicionar autenticação (API Keys, JWT) em produção

---

## 📞 Suporte

Para mais informações ou reportar problemas:

- **Autor**: Renan Santos Mendes
- **Email**: renansantosmendes@gmail.com
- **GitHub**: https://github.com/renansantosmendes

---

**Última atualização**: Março 2026
