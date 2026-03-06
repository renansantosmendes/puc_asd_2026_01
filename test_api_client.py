"""Cliente para testar a API"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Testar health check."""
    print("\n[1] Testando Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Testar informações do modelo."""
    print("\n[2] Testando Model Info...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Testar predição única."""
    print("\n[3] Testando Predição Única...")
    
    # Features aleatórias (21 features)
    features = [120.0, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01,
                0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0, 0.0]
    
    payload = {"features": features}
    response = requests.post(f"{BASE_URL}/predictions/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Testar predição em batch."""
    print("\n[4] Testando Predição em Batch...")
    
    samples = [
        [120.0, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01,
         0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0, 0.0],
        [125.0, 0.6, 0.35, 0.25, 0.15, 0.09, 0.06, 0.04, 0.03, 0.02,
         0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0],
        [118.0, 0.4, 0.25, 0.15, 0.05, 0.07, 0.04, 0.02, 0.01, 0.005,
         0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0, 0.0, 0.0]
    ]
    
    payload = {"samples": samples}
    response = requests.post(f"{BASE_URL}/predictions/predict-batch", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Executar todos os testes."""
    print("=" * 60)
    print("🧪 Testando Fetal Health Prediction API")
    print("=" * 60)
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "=" * 60)
        print("✓ Todos os testes concluídos!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando: python run_api.py")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
