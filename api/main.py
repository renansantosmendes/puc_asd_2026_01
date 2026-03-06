from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# ============================================================================
# Configuração do FastAPI
# ============================================================================

app = FastAPI(
    title="🏥 API de Classificação de Saúde Fetal",
    description="""
    API para classificação de saúde fetal usando Gradient Boosting.
    
    ## 📋 Sobre o Modelo
    
    Este modelo foi treinado com cardiotocografias (CTGs) para classificar a saúde fetal em 3 categorias:
    
    - **Classe 1**: Normal
    - **Classe 2**: Suspeito
    - **Classe 3**: Patológico
    
    O modelo utiliza 21 características extraídas de exames de CTG para fazer as previsões.
    
    ## 🔐 Configuração
    
    A API carrega credenciais do DagsHub/MLflow a partir do arquivo `.env`:
    - `MLFLOW_TRACKING_USERNAME`
    - `MLFLOW_TRACKING_PASSWORD`
    - `MLFLOW_TRACKING_URI`
    
    ## 📞 Endpoints Disponíveis
    
    - `GET /health` - Verifica saúde da API e disponibilidade do modelo
    - `POST /predict` - Faz predição de saúde fetal
    """,
    version="1.0.0",
    authors=[
        {
            "name": "Renan Santos Mendes",
            "email": "renansantosmendes@gmail.com"
        }
    ]
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================================
# Modelos Pydantic para Documentação
# ============================================================================

class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""
    status: str = Field(..., description="Status da API (ok/error)")
    model_loaded: bool = Field(..., description="Se o modelo foi carregado com sucesso")


class PredictionResponse(BaseModel):
    """Resposta do endpoint de predição."""
    prediction: int = Field(..., description="Classe predita (1=Normal, 2=Suspeito, 3=Patológico)", example=1)
    confidence: float = Field(..., description="Confiança da predição (0-1)", example=0.95)
    probabilities: dict = Field(
        ..., 
        description="Probabilidades para cada classe",
        example={"1": 0.95, "2": 0.04, "3": 0.01}
    )

# ============================================================================
# Variáveis Globais
# ============================================================================

model = None
scaler = None

# Nomes das 21 características esperadas
FEATURE_NAMES = [
    "baseline", "acceleration", "fetal_movement", "uterine_contractions",
    "severe_decelerations", "prolonged_decelerations", "abnormal_short_term_variability",
    "mean_value_of_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
    "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
    "histogram_tendency", "fetal_health"
]


# ============================================================================
# Evento de Startup - Carregar Modelo do MLflow
# ============================================================================

@app.on_event("startup")
async def startup():
    """
    Evento executado ao iniciar a aplicação.
    Carrega o modelo treinado do MLflow/DagsHub.
    """
    global model, scaler
    
    print("\n" + "="*70)
    print("🚀 Iniciando API de Classificação de Saúde Fetal")
    print("="*70)
    
    # Carregar variáveis de ambiente
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip()
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip()
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
    
    if not all([mlflow_username, mlflow_password, mlflow_uri]):
        print("❌ Erro: Variáveis de ambiente do MLflow não configuradas no arquivo .env")
        print("   Configure MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD e MLFLOW_TRACKING_URI")
        print("="*70 + "\n")
        return
    
    try:
        # Configurar variáveis de ambiente
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
        mlflow.set_tracking_uri(mlflow_uri)
        
        print(f"✅ MLflow configurado com sucesso")
        print(f"   URI: {mlflow_uri}")
        print(f"   Usuário: {mlflow_username}")
        
        # Carregar modelo do MLflow
        print(f"\n⏳ Buscando modelos registrados no MLflow...")
        client = MlflowClient()
        
        try:
            logged_models = client.search_logged_models(['0'])
        except Exception as search_error:
            print(f"⚠️  Erro ao buscar modelos: {str(search_error)}")
            print(f"   Verifique se:")
            print(f"   1. As credenciais do DagsHub estão corretas")
            print(f"   2. A URI do MLflow está correta: {mlflow_uri}")
            print(f"   3. Você tem acesso à internet e ao DagsHub")
            print("="*70 + "\n")
            return
        
        if logged_models and len(logged_models) > 0:
            print(f"✅ {len(logged_models)} modelo(s) encontrado(s)")
            
            model_uri = logged_models[0].model_uri
            print(f"⏳ Carregando modelo: {model_uri}")
            
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Modelo carregado com sucesso!")
            except Exception as load_error:
                print(f"❌ Erro ao carregar o modelo: {str(load_error)}")
                print(f"   O modelo pode estar corrompido ou a conexão falhou")
                print("="*70 + "\n")
                return
        else:
            print("⚠️  Aviso: Nenhum modelo encontrado no MLflow (experimento 0)")
            print("   Certifique-se de que:")
            print("   1. O modelo foi registrado corretamente no MLflow")
            print("   2. Está no experimento com ID '0'")
            print("   3. Seu repositório contém os modelos")
            model = None
        
        # Inicializar scaler
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame([[0]*21]))
        print(f"✅ Scaler inicializado com sucesso")
        
        print("="*70)
        print("✅ API pronta para receber requisições!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Erro inesperado ao inicializar a API: {str(e)}")
        print(f"   Tipo de erro: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")


# ============================================================================
# Endpoints da API
# ============================================================================

@app.get(
    "/health",
    tags=["Health Check"],
    response_model=HealthResponse,
    summary="Verificar Saúde da API",
    description="""
    Endpoint para verificar se a API está funcionando e se o modelo foi carregado com sucesso.
    
    **Retorna:**
    - `status`: Estado da API (sempre 'ok' se responder)
    - `model_loaded`: Se o modelo de classificação foi carregado do MLflow
    
    **Exemplo de uso:**
    ```bash
    curl -X GET "http://localhost:8000/health"
    ```
    """
)
async def health() -> HealthResponse:
    """Verifica o status da API e disponibilidade do modelo."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post(
    "/predict",
    tags=["Predições"],
    response_model=PredictionResponse,
    summary="Fazer Predição de Saúde Fetal",
    description="""
    Realiza a classificação de saúde fetal baseado em características de cardiotocografia.
    
    ## 📝 Parâmetros
    
    O endpoint espera um JSON com um array de 21 valores float representando as características:
    
    1. `baseline` - Frequência cardíaca basal fetal
    2. `acceleration` - Acelerações
    3. `fetal_movement` - Movimentos fetais
    4. `uterine_contractions` - Contrações uterinas
    5. `severe_decelerations` - Desacelerações severas
    6. `prolonged_decelerations` - Desacelerações prolongadas
    7. `abnormal_short_term_variability` - Variabilidade anormal curto prazo
    8. `mean_value_of_short_term_variability` - Valor médio variabilidade curto prazo
    9. `percentage_of_time_with_abnormal_long_term_variability` - Percentual tempo com variabilidade longa anormal
    10. `mean_value_of_long_term_variability` - Valor médio variabilidade longo prazo
    11. `histogram_width` - Largura do histograma
    12. `histogram_min` - Mínimo do histograma
    13. `histogram_max` - Máximo do histograma
    14. `histogram_number_of_peaks` - Número de picos no histograma
    15. `histogram_number_of_zeroes` - Número de zeros no histograma
    16. `histogram_mode` - Modo do histograma
    17. `histogram_mean` - Média do histograma
    18. `histogram_median` - Mediana do histograma
    19. `histogram_variance` - Variância do histograma
    20. `histogram_tendency` - Tendência do histograma
    21. `fetal_health` - Saúde fetal (não usado na predição, deixar como 0)
    
    ## 📊 Retorno
    
    - `prediction`: Classe predita (1=Normal, 2=Suspeito, 3=Patológico)
    - `confidence`: Confiança da predição entre 0 e 1
    - `probabilities`: Probabilidades para cada classe
    
    ## 💡 Exemplo de uso
    
    ```bash
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{
        "features": [120, 0.5, 0.2, 0.1, 0, 0, 0.1, 0.5, 0.05, 0.3, 30, 100, 160, 5, 3, 130, 140, 135, 50, 0, 0]
      }'
    ```
    
    ## 🔍 Interpretação da Resposta
    
    - **Classe 1 (Normal)**: Predição segura, pode prosseguir com acompanhamento regular
    - **Classe 2 (Suspeito)**: Requer acompanhamento mais próximo, possível intervenção
    - **Classe 3 (Patológico)**: Requer intervenção imediata, situação crítica
    
    A `confidence` indica o quão segura é a predição (valores mais altos = mais seguros).
    """
)
async def predict(
    features: list[float] = Query(
        ...,
        description="Lista de 21 características de cardiotocografia",
        min_items=21,
        max_items=21
    )
) -> PredictionResponse:
    """
    Realiza predição de saúde fetal a partir de características de CTG.
    
    Args:
        features: Lista com exatamente 21 valores float
        
    Returns:
        PredictionResponse com predição, confiança e probabilidades
    """
    if model is None:
        return {
            "error": "❌ Modelo não foi carregado. Verifique as configurações do MLflow e tente novamente.",
            "prediction": None,
            "confidence": None,
            "probabilities": None
        }
    
    if scaler is None:
        return {
            "error": "❌ Scaler não foi inicializado. Tente reiniciar a API.",
            "prediction": None,
            "confidence": None,
            "probabilities": None
        }
    
    try:
        # Validar que todas as features são números
        features_array = np.array(features, dtype=float)
        
        # Normalizar features
        X = features_array.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Fazer predição
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        return PredictionResponse(
            prediction=int(pred),
            confidence=float(np.max(proba)),
            probabilities={
                "1": float(proba[0]),
                "2": float(proba[1]),
                "3": float(proba[2])
            }
        )
    
    except ValueError as ve:
        return {
            "error": f"❌ Erro de validação: As features devem ser valores numéricos. Detalhes: {str(ve)}",
            "prediction": None,
            "confidence": None,
            "probabilities": None
        }
    except Exception as e:
        return {
            "error": f"❌ Erro ao fazer predição: {str(e)}. Tipo de erro: {type(e).__name__}",
            "prediction": None,
            "confidence": None,
            "probabilities": None
        }
