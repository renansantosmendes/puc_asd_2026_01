"""
Presentation Layer: CLI

Interface de linha de comando para o aplicativo.
Coordena os use cases e exibe resultados ao usuário.
"""

from typing import Optional
import pandas as pd
from src.application.use_cases.load_data_use_case import LoadDataUseCase
from src.application.use_cases.prepare_data_use_case import PrepareDataUseCase
from src.application.use_cases.train_model_use_case import TrainModelUseCase
from src.application.use_cases.evaluate_model_use_case import EvaluateModelUseCase
from src.application.dto.model_dto import (
    LoadDataRequest, PrepareDataRequest, TrainModelRequest, EvaluateModelRequest
)
from src.infrastructure.repositories.data_repository import CSVDataRepository
from src.infrastructure.services.preprocessing_service import SKLearnPreprocessingService
from src.infrastructure.services.model_training_service import ScikitLearnModelTrainingService
from src.infrastructure.services.evaluation_service import MetricsEvaluationService
from src.infrastructure.services.visualization_service import MatplotlibVisualizationService
from src.infrastructure.services.persistence_service import PickleModelRepository


class FetalHealthCLI:
    """Interface CLI para classificação de saúde fetal."""
    
    def __init__(self):
        """Inicializa a CLI e seus dependências."""
        # Repositórios e Serviços
        self.data_repository = CSVDataRepository()
        self.preprocessing_service = SKLearnPreprocessingService()
        self.training_service = ScikitLearnModelTrainingService()
        self.evaluation_service = MetricsEvaluationService()
        self.visualization_service = MatplotlibVisualizationService()
        self.persistence_service = PickleModelRepository()
        
        # Use Cases
        self.load_data_use_case = LoadDataUseCase(self.data_repository)
        self.prepare_data_use_case = PrepareDataUseCase(self.preprocessing_service)
        self.train_model_use_case = TrainModelUseCase(self.training_service)
        self.evaluate_model_use_case = EvaluateModelUseCase(self.evaluation_service)
    
    def run(self, data_path: str):
        """
        Executa o pipeline completo.
        
        Args:
            data_path: Caminho para o arquivo CSV
        """
        print("=" * 70)
        print("CLASSIFICAÇÃO DE SAÚDE FETAL - PIPELINE ML")
        print("=" * 70)
        
        # 1. Carregar dados
        print("\n[1/6] Carregando dados...")
        load_response = self.load_data_use_case.execute(
            LoadDataRequest(path=data_path)
        )
        
        if not load_response.success:
            print(f"❌ Erro: {load_response.error}")
            return
        
        print(f"✓ {load_response.message}")
        data = load_response.data
        
        # Separar features e target
        X, y = self.data_repository.get_features_and_target(data)
        
        # 2. Visualizar distribuição
        print("\n[2/6] Visualizando distribuição das classes...")
        data_copy = data.copy()
        self.visualization_service.plot_class_distribution(
            data_copy,
            "Distribuição das Classes - Dataset Completo"
        )
        
        # 3. Preparar dados
        print("\n[3/6] Preparando dados...")
        prepare_response = self.prepare_data_use_case.execute(
            PrepareDataRequest(X=X, y=y, test_size=0.3, stratify=True)
        )
        
        if not prepare_response.success:
            print(f"❌ Erro: {prepare_response.error}")
            return
        
        print("✓ Dados preparados com sucesso")
        print(f"  - Treino: {prepare_response.X_train.shape}")
        print(f"  - Teste: {prepare_response.X_test.shape}")
        
        X_train = prepare_response.X_train
        X_test = prepare_response.X_test
        y_train = prepare_response.y_train
        y_test = prepare_response.y_test
        sample_weights = prepare_response.sample_weights
        
        # Visualizar distribuição nos conjuntos
        y_train_df = y_train.to_frame()
        self.visualization_service.plot_class_distribution(
            y_train_df,
            "Distribuição - Conjunto de Treino"
        )
        
        y_test_df = y_test.to_frame()
        self.visualization_service.plot_class_distribution(
            y_test_df,
            "Distribuição - Conjunto de Teste"
        )
        
        # 4. Treinar modelo
        print("\n[4/6] Treinando Gradient Boosting...")
        train_response = self.train_model_use_case.execute(
            TrainModelRequest(
                model_type='gradient_boosting',
                X_train=X_train,
                y_train=y_train,
                sample_weights=sample_weights
            )
        )
        
        if not train_response.success:
            print(f"❌ Erro: {train_response.error}")
            return
        
        model = train_response.model
        print(f"✓ Modelo treinado em {train_response.training_time:.2f}s")
        
        # 5. Avaliar modelo
        print("\n[5/6] Avaliando modelo...")
        eval_response = self.evaluate_model_use_case.execute(
            EvaluateModelRequest(
                model=model,
                X_test=X_test,
                y_test=y_test
            )
        )
        
        if not eval_response.success:
            print(f"❌ Erro: {eval_response.error}")
            return
        
        print(f"✓ Acurácia: {eval_response.accuracy:.4f}")
        print("\n" + eval_response.classification_report)
        
        # Visualizar matriz de confusão
        self.visualization_service.plot_confusion_matrix(
            model, X_test, y_test,
            "Matriz de Confusão - Gradient Boosting"
        )
        
        # 6. Salvar modelo
        print("\n[6/6] Salvando modelo...")
        self.persistence_service.save_model(model, "asd_model.pkl")
        print("✓ Modelo salvo em 'asd_model.pkl'")
        
        print("\n" + "=" * 70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
