"""
exemplo_api_test.py

Exemplo simples de como usar a API de classificação de saúde fetal.

Certifique-se de que:
1. O arquivo .env está configurado com as credenciais do MLflow/DagsHub
2. A API está rodando (execute: python run_api.py)
3. As dependências estão instaladas (pip install requests)

Uso:
    python example_api_test.py
"""

import requests
import json
import sys


def test_health():
    """Testa o endpoint de health check."""
    print("\n" + "="*70)
    print("🏥 Testando Health Check")
    print("="*70)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print("\n✅ Health Check realizado com sucesso!")
        print(f"   Status: {data['status']}")
        print(f"   Modelo carregado: {'✅ Sim' if data['model_loaded'] else '❌ Não'}")
        
        if not data['model_loaded']:
            print("\n⚠️  Aviso: O modelo não foi carregado!")
            print("   Verifique:")
            print("   1. Arquivo .env está configurado?")
            print("   2. Credenciais do DagsHub estão corretas?")
            print("   3. O modelo existe no MLflow?")
        
        return data['model_loaded']
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar à API")
        print("   Certifique-se de que a API está rodando (execute: python run_api.py)")
        return False
    except Exception as e:
        print(f"\n❌ Erro ao testar health check: {str(e)}")
        return False


def test_prediction():
    """Testa o endpoint de predição com exemplos de diferentes cenários."""
    
    print("\n" + "="*70)
    print("🔮 Testando Predições")
    print("="*70)
    
    # Exemplo 1: Caso Normal (esperado retornar classe 1)
    example_normal = {
        "name": "Caso Normal",
        "features": [
            120.0,   # baseline
            0.5,     # acceleration
            0.2,     # fetal_movement
            0.1,     # uterine_contractions
            0.0,     # severe_decelerations
            0.0,     # prolonged_decelerations
            0.1,     # abnormal_short_term_variability
            0.5,     # mean_value_of_short_term_variability
            0.05,    # percentage_of_time_with_abnormal_long_term_variability
            0.3,     # mean_value_of_long_term_variability
            30.0,    # histogram_width
            100.0,   # histogram_min
            160.0,   # histogram_max
            5.0,     # histogram_number_of_peaks
            3.0,     # histogram_number_of_zeroes
            130.0,   # histogram_mode
            140.0,   # histogram_mean
            135.0,   # histogram_median
            50.0,    # histogram_variance
            0.0,     # histogram_tendency
            0.0      # fetal_health
        ]
    }
    
    # Exemplo 2: Caso Suspeito
    example_suspeito = {
        "name": "Caso Suspeito",
        "features": [
            110.0,   # baseline (ligeiramente reduzido)
            0.3,     # acceleration (reduzido)
            0.15,    # fetal_movement (reduzido)
            0.2,     # uterine_contractions (aumentado)
            0.05,    # severe_decelerations (presente)
            0.02,    # prolonged_decelerations (presente)
            0.2,     # abnormal_short_term_variability (aumentado)
            0.6,     # mean_value_of_short_term_variability
            0.15,    # percentage_of_time_with_abnormal_long_term_variability (aumentado)
            0.5,     # mean_value_of_long_term_variability (aumentado)
            40.0,    # histogram_width (aumentado)
            90.0,    # histogram_min
            150.0,   # histogram_max
            4.0,     # histogram_number_of_peaks
            5.0,     # histogram_number_of_zeroes
            120.0,   # histogram_mode
            130.0,   # histogram_mean
            125.0,   # histogram_median
            60.0,    # histogram_variance
            0.1,     # histogram_tendency
            0.0      # fetal_health
        ]
    }
    
    # Exemplo 3: Caso Patológico
    example_patologico = {
        "name": "Caso Patológico",
        "features": [
            100.0,   # baseline (reduzido significativamente)
            0.1,     # acceleration (muito reduzido)
            0.05,    # fetal_movement (muito reduzido)
            0.4,     # uterine_contractions (muito aumentado)
            0.2,     # severe_decelerations (muito presente)
            0.15,    # prolonged_decelerations (muito presente)
            0.5,     # abnormal_short_term_variability (muito aumentado)
            0.8,     # mean_value_of_short_term_variability
            0.4,     # percentage_of_time_with_abnormal_long_term_variability (muito aumentado)
            0.8,     # mean_value_of_long_term_variability (muito aumentado)
            60.0,    # histogram_width (muito aumentado)
            70.0,    # histogram_min
            140.0,   # histogram_max
            2.0,     # histogram_number_of_peaks
            10.0,    # histogram_number_of_zeroes
            100.0,   # histogram_mode
            110.0,   # histogram_mean
            105.0,   # histogram_median
            80.0,    # histogram_variance
            0.5,     # histogram_tendency
            0.0      # fetal_health
        ]
    }
    
    examples = [example_normal, example_suspeito, example_patologico]
    
    for example in examples:
        print(f"\n📊 Testando: {example['name']}")
        print("-" * 70)
        
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": example['features']},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Interpretar resultado
            class_map = {1: "🟢 Normal", 2: "🟡 Suspeito", 3: "🔴 Patológico"}
            prediction_text = class_map.get(result['prediction'], "Desconhecido")
            
            print(f"✅ Predição realizada com sucesso!")
            print(f"   Classe: {prediction_text}")
            print(f"   Confiança: {result['confidence']:.2%}")
            print(f"   Probabilidades:")
            print(f"      - Normal: {result['probabilities']['1']:.2%}")
            print(f"      - Suspeito: {result['probabilities']['2']:.2%}")
            print(f"      - Patológico: {result['probabilities']['3']:.2%}")
        
        except requests.exceptions.ConnectionError:
            print("❌ Erro: Não foi possível conectar à API")
            print("   Certifique-se de que a API está rodando")
            return False
        except Exception as e:
            print(f"❌ Erro ao fazer predição: {str(e)}")
            return False
    
    return True


def main():
    """Função principal."""
    print("\n" + "="*70)
    print("🏥 Teste da API de Classificação de Saúde Fetal")
    print("="*70)
    
    # Teste 1: Health Check
    model_loaded = test_health()
    
    if not model_loaded:
        print("\n⚠️  Não é possível testar predições sem o modelo carregado")
        sys.exit(1)
    
    # Teste 2: Predições
    success = test_prediction()
    
    if success:
        print("\n" + "="*70)
        print("✅ Todos os testes foram realizados com sucesso!")
        print("="*70)
        print("\n💡 Dica: Acesse http://localhost:8000/docs para documentação interativa")
    else:
        print("\n" + "="*70)
        print("❌ Alguns testes falharam")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
