"""
Saúde Fetal - Modelo de Aprendizado de Máquina para Classificação

Autor: Renan Santos Mendes
Email: renansantosmendes@gmail.com

Descrição: Este script apresenta um exemplo de modelo de aprendizado de máquina 
para um problema de classificação.

As Cardiotocografias (CTGs) são opções simples e de baixo custo para avaliar a saúde fetal,
permitindo que os profissionais de saúde atuem na prevenção da mortalidade infantil e materna.
O próprio equipamento funciona enviando pulsos de ultrassom e lendo sua resposta, lançando luz 
sobre a frequência cardíaca fetal (FCF), movimentos fetais, contrações uterinas e muito mais.

Este conjunto de dados contém 2126 registros de características extraídas de exames de 
Cardiotocografias, que foram então classificados por três obstetras especialistas em 3 classes:
- Normal
- Suspeito
- Patológico
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Importar MLflow (opcional)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow não está instalado. Pulando integração com MLflow.")


# ============================================================================
# 1 - IMPORTAÇÃO E LEITURA DO DATASET
# ============================================================================

def load_data():
    """Carrega o dataset de saúde fetal."""
    print("Carregando dataset...")
    data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')
    return data


# ============================================================================
# 2 - PREPARAÇÃO DOS DADOS
# ============================================================================

def prepare_data(data):
    """
    Prepara os dados antes do treinamento do modelo.
    
    Separação de features e target
    Normalização usando StandardScaler
    """
    print("\nPreparando dados...")
    
    # Separar features (X) e target (y)
    X = data.iloc[:, :-1]
    y = data["fetal_health"]
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=list(X.columns))
    
    return X_df, y, scaler


def split_data(X_df, y, test_size=0.3, stratify=True, random_state=42):
    """
    Divide os dados em treino e teste com estratificação opcional.
    """
    print(f"\nDividindo dados (test_size={test_size})...")
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y,
            test_size=test_size,
            random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test


def compute_class_weights(y_train):
    """
    Calcula pesos para cada classe para lidar com desequilíbrio.
    """
    print("\nCalculando pesos das classes...")
    
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    print("Pesos calculados para cada classe:")
    for cls, weight in class_weights_dict.items():
        print(f"  Classe {cls}: {weight:.4f}")
    
    # Criar array de sample_weights
    sample_weights = np.array([class_weights_dict[cls] for cls in y_train])
    
    return class_weights_dict, sample_weights


# ============================================================================
# 3 - VISUALIZAÇÃO
# ============================================================================

def plot_fetal_health_distribution(df, title="Distribuição das Classes de Saúde Fetal"):
    """
    Gera um gráfico de barras da distribuição das classes de saúde fetal.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='fetal_health', data=df, palette='viridis', hue='fetal_health', legend=False)
    
    plt.title(title)
    plt.xlabel('Saúde Fetal (1: Normal, 2: Suspeito, 3: Patológico)')
    plt.ylabel('Quantidade')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, title="Matriz de Confusão"):
    """
    Plota a matriz de confusão do modelo.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Normal', 'Suspeito', 'Patológico'],
        ax=ax, cmap='Blues'
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4 - TREINAMENTO DE MODELOS
# ============================================================================

def train_decision_tree(X_train, y_train):
    """
    Treina um modelo Decision Tree.
    """
    print("\n=== Treinando Decision Tree ===")
    start = time()
    
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    
    elapsed = time() - start
    print(f"Tempo de treinamento: {elapsed:.4f}s")
    
    return tree_clf


def train_gradient_boosting(X_train, y_train, sample_weights=None, 
                           max_depth=7, n_estimators=200, learning_rate=0.001,
                           random_state=42):
    """
    Treina um modelo Gradient Boosting.
    """
    print("\n=== Treinando Gradient Boosting ===")
    start = time()
    
    grd_clf = GradientBoostingClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    if sample_weights is not None:
        grd_clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grd_clf.fit(X_train, y_train)
    
    elapsed = time() - start
    print(f"Tempo de treinamento: {elapsed:.4f}s")
    
    return grd_clf


def grid_search_optimization(X_train, y_train, param_grid=None):
    """
    Otimiza hiperparâmetros usando GridSearchCV.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01],
            'max_depth': [10, 15]
        }
    
    print("\n=== Iniciando Grid Search ===")
    start = time()
    
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    print("Processando...")
    grid_search.fit(X_train, y_train)
    
    elapsed = time() - start
    print(f"Tempo de Grid Search: {elapsed:.4f}s")
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor acurácia no treino: {grid_search.best_score_:.4f}")
    
    return grid_search


# ============================================================================
# 5 - AVALIAÇÃO
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo nos dados de teste.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAcurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'Suspeito', 'Patológico']))
    
    return y_pred


# ============================================================================
# 6 - SALVAMENTO E CARREGAMENTO DO MODELO
# ============================================================================

def save_model(model, filepath="model.pkl"):
    """
    Salva o modelo em arquivo pickle.
    """
    print(f"\nSalvando modelo em {filepath}...")
    pickle.dump(model, open(filepath, 'wb'))
    print("Modelo salvo com sucesso!")


def load_model(filepath="model.pkl"):
    """
    Carrega um modelo de arquivo pickle.
    """
    print(f"Carregando modelo de {filepath}...")
    model = pickle.load(open(filepath, 'rb'))
    print("Modelo carregado com sucesso!")
    return model


# ============================================================================
# 7 - MLFLOW (Opcional)
# ============================================================================

def setup_mlflow():
    """
    Configura integração com MLflow e DagsHub usando variáveis do arquivo .env
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow não disponível. Pulando setup.")
        return False
    
    print("\nConfigurando MLflow...")
    
    # Carregar variáveis de ambiente
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    
    if not all([mlflow_username, mlflow_password, mlflow_uri]):
        print("Erro: Variáveis de ambiente do MLflow não configuradas no arquivo .env")
        return False
    
    # Configurar variáveis de ambiente
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.sklearn.autolog(log_input_examples=True)
    
    print(f"✓ MLflow configurado com sucesso para: {mlflow_uri}")
    return True


def get_logged_models(client, experiment_id='0'):
    """
    Recupera modelos registrados no MLflow.
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    logged_models = client.search_logged_models([experiment_id])
    return logged_models


def load_mlflow_model(model_uri):
    """
    Carrega modelo do MLflow.
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    return mlflow.pyfunc.load_model(model_uri)


# ============================================================================
# MAIN - FLUXO PRINCIPAL
# ============================================================================

def main():
    """
    Executa o fluxo principal de treinamento e avaliação.
    """
    print("=" * 70)
    print("CLASSIFICAÇÃO DE SAÚDE FETAL - MACHINE LEARNING")
    print("=" * 70)
    
    # 1. Carregar dados
    data = load_data()
    print(f"Dataset carregado: {data.shape[0]} amostras, {data.shape[1]} features")
    print("\nPrimeiras linhas do dataset:")
    print(data.head())
    
    # 2. Preparar dados
    X_df, y, scaler = prepare_data(data)
    print(f"\nDados normalizados:")
    print(X_df.head())
    
    # 3. Dividir dados
    X_train, X_test, y_train, y_test = split_data(X_df, y, stratify=True)
    print(f"\nTamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de teste: {X_test.shape}")
    
    # 4. Visualizar distribuição
    print("\nDistribuição das classes no dataset completo:")
    print(y.value_counts())
    
    data_copy = data.copy()
    plot_fetal_health_distribution(data_copy, "Distribuição das Classes - Dataset Completo")
    
    y_train_df = y_train.to_frame()
    plot_fetal_health_distribution(y_train_df, "Distribuição das Classes - Conjunto de Treino")
    
    y_test_df = y_test.to_frame()
    plot_fetal_health_distribution(y_test_df, "Distribuição das Classes - Conjunto de Teste")
    
    # 5. Calcular pesos das classes
    class_weights_dict, sample_weights = compute_class_weights(y_train)
    
    # 6. Treinar Decision Tree
    tree_clf = train_decision_tree(X_train, y_train)
    y_pred_tree = evaluate_model(tree_clf, X_test, y_test)
    
    # 7. Treinar Gradient Boosting
    grd_clf = train_gradient_boosting(X_train, y_train, sample_weights=sample_weights)
    y_pred_gb = evaluate_model(grd_clf, X_test, y_test)
    plot_confusion_matrix(grd_clf, X_test, y_test, "Matriz de Confusão - Gradient Boosting Inicial")
    
    # 8. Grid Search para otimização
    grid_search = grid_search_optimization(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_best = evaluate_model(best_model, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, "Matriz de Confusão - Melhor Modelo (Grid Search)")
    
    # 9. Salvar modelos
    save_model(grd_clf, "asd_o9.pkl")
    
    # 10. Testar carregamento
    loaded_model = load_model("asd_o9.pkl")
    print(f"\nModelo carregado: {loaded_model}")
    predictions = loaded_model.predict(X_test)
    print(f"Predições do modelo carregado (primeiras 5): {predictions[:5]}")
    
    # 11. MLflow (opcional)
    if MLFLOW_AVAILABLE:
        setup_mlflow()
        
        # Treinar modelo com MLflow logging
        print("\n=== Treinando modelo com MLflow logging ===")
        grd_clf_final = train_gradient_boosting(
            X_train, y_train,
            sample_weights=sample_weights,
            max_depth=7,
            n_estimators=250,
            learning_rate=0.001
        )
        
        # Recuperar modelos registrados
        client = MlflowClient()
        logged_models = get_logged_models(client)
        if logged_models:
            print(f"\nModelos registrados: {len(logged_models)}")
            model_uri = logged_models[0].model_uri
            print(f"URI do primeiro modelo: {model_uri}")
            
            # Carregar modelo do MLflow
            mlflow_model = load_mlflow_model(model_uri)
            if mlflow_model:
                mlflow_predictions = mlflow_model.predict(X_test)
                print(f"Predições do MLflow (primeiras 5): {mlflow_predictions[:5]}")
    
    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
