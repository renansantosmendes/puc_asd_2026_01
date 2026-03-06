#!/usr/bin/env python3
"""
check_setup.py

Script de diagnóstico para verificar se a API está configurada corretamente.

Uso:
    python check_setup.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def print_header(title):
    """Imprime um cabeçalho formatado."""
    print("\n" + "="*70)
    print(f"🔍 {title}")
    print("="*70)

def check_env_file():
    """Verifica se o arquivo .env existe e está configurado."""
    print_header("Verificando Arquivo .env")
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ Arquivo .env não encontrado!")
        print("   Crie um arquivo .env na raiz do projeto com:")
        print("   MLFLOW_TRACKING_USERNAME=seu_usuario")
        print("   MLFLOW_TRACKING_PASSWORD=seu_token")
        print("   MLFLOW_TRACKING_URI=https://dagshub.com/seu_usuario/seu_repo.mlflow")
        return False
    
    print("✅ Arquivo .env encontrado")
    return True

def check_env_variables():
    """Verifica se as variáveis de ambiente estão configuradas."""
    print_header("Verificando Variáveis de Ambiente")
    
    load_dotenv()
    
    username = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip()
    password = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip()
    uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
    
    all_present = True
    
    # Verificar username
    if username:
        print(f"✅ MLFLOW_TRACKING_USERNAME = {username}")
    else:
        print("❌ MLFLOW_TRACKING_USERNAME não está definido")
        all_present = False
    
    # Verificar password
    if password:
        password_masked = '*' * (len(password) - 4) + password[-4:]
        print(f"✅ MLFLOW_TRACKING_PASSWORD = {password_masked}")
    else:
        print("❌ MLFLOW_TRACKING_PASSWORD não está definido")
        all_present = False
    
    # Verificar URI
    if uri:
        print(f"✅ MLFLOW_TRACKING_URI = {uri}")
        
        # Validar formato da URI
        if "dagshub.com" not in uri:
            print("⚠️  Aviso: URI não contém 'dagshub.com'")
        if not uri.endswith(".mlflow"):
            print("⚠️  Aviso: URI não termina com '.mlflow'")
        
        # Extrair repositório da URI
        try:
            parts = uri.split("/")
            if len(parts) >= 4:
                repo = parts[-1].replace(".mlflow", "")
                print(f"   Repositório: {repo}")
        except:
            pass
    else:
        print("❌ MLFLOW_TRACKING_URI não está definido")
        all_present = False
    
    return all_present

def check_dependencies():
    """Verifica se as dependências estão instaladas."""
    print_header("Verificando Dependências")
    
    required_packages = {
        'fastapi': 'FastAPI (web framework)',
        'uvicorn': 'Uvicorn (ASGI server)',
        'mlflow': 'MLflow (model registry)',
        'sklearn': 'Scikit-Learn (ML library)',
        'numpy': 'NumPy (numerical computing)',
        'pandas': 'Pandas (data manipulation)',
        'dotenv': 'Python-dotenv (environment variables)',
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} não está instalado")
            all_installed = False
    
    if not all_installed:
        print("\n💡 Para instalar as dependências, execute:")
        print("   pip install -r requirements.txt")
    
    return all_installed

def check_mlflow_connection():
    """Tenta conectar ao MLflow usando as credenciais."""
    print_header("Testando Conexão com MLflow/DagsHub")
    
    load_dotenv()
    
    username = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip()
    password = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip()
    uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
    
    if not all([username, password, uri]):
        print("⏭️  Pulando teste de conexão (variáveis não configuradas)")
        return False
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Configurar MLflow
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        mlflow.set_tracking_uri(uri)
        
        print(f"⏳ Conectando a {uri}...")
        
        # Tentar buscar modelos
        client = MlflowClient()
        logged_models = client.search_logged_models(['0'])
        
        print(f"✅ Conexão com MLflow bem-sucedida")
        print(f"   Modelos encontrados no experimento 0: {len(logged_models)}")
        
        if logged_models:
            print(f"✅ Modelo disponível para uso")
            print(f"   URI do modelo: {logged_models[0].model_uri}")
            return True
        else:
            print("⚠️  Aviso: Nenhum modelo encontrado no experimento 0")
            print("   Certifique-se de que o modelo foi registrado no MLflow")
            return False
    
    except ImportError:
        print("⏭️  MLflow não está instalado (pulando teste de conexão)")
        return False
    except Exception as e:
        print(f"❌ Erro ao conectar ao MLflow: {str(e)}")
        print(f"   Verifique:")
        print(f"   1. Credenciais do DagsHub estão corretas?")
        print(f"   2. URI do MLflow está correta?")
        print(f"   3. Você tem acesso à internet e ao DagsHub?")
        return False

def check_python_version():
    """Verifica a versão do Python."""
    print_header("Verificando Versão do Python")
    
    version = sys.version_info
    version_string = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Versão do Python: {version_string}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Versão suportada")
        return True
    else:
        print("❌ Python 3.8+ é necessário")
        return False

def main():
    """Executa todos os testes."""
    print("\n" + "="*70)
    print("🏥 Verificação de Configuração - API de Saúde Fetal")
    print("="*70)
    
    results = {
        "Python": check_python_version(),
        ".env": check_env_file(),
        "Variáveis": check_env_variables() if check_env_file() else False,
        "Dependências": check_dependencies(),
        "MLflow": check_mlflow_connection(),
    }
    
    # Resumo final
    print_header("Resumo da Verificação")
    
    for check, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ Tudo está configurado corretamente!")
        print("   Você pode iniciar a API com: python run_api.py")
        print("="*70)
        return 0
    else:
        print("❌ Há problemas na configuração")
        print("   Consulte o arquivo TROUBLESHOOTING.md para soluções")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
