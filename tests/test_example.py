"""
Exemplo de Testes

Este arquivo contém exemplos de como estruturar testes usando pytest.
Segue a arquitetura de camadas do projeto.

Para rodar os testes:
    pytest tests/
    pytest tests/ -v  # verbose
    pytest tests/ --cov=src  # com coverage
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.domain.entities.model import FetalHealthDataset, TrainTestSplit
from src.infrastructure.repositories.data_repository import CSVDataRepository
from src.infrastructure.services.preprocessing_service import SKLearnPreprocessingService
from src.application.use_cases.load_data_use_case import LoadDataUseCase
from src.application.dto.model_dto import LoadDataRequest


class TestEntities:
    """Testes das entidades de domínio."""
    
    def test_fetal_health_dataset_creation(self):
        """Testa criação de FetalHealthDataset."""
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2, 1])
        feature_names = ['feature1']
        
        dataset = FetalHealthDataset(X=X, y=y, feature_names=feature_names)
        
        assert dataset.X.shape == (3, 1)
        assert len(dataset.y) == 3
        assert dataset.feature_names == feature_names


class TestDataRepository:
    """Testes do repositório de dados."""
    
    def test_load_data_success(self):
        """Testa carregamento bem-sucedido de dados."""
        # Mock do pandas.read_csv
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [1, 2, 1]
        })
        
        repo = CSVDataRepository()
        
        with patch('pandas.read_csv', return_value=mock_df):
            result = repo.load_data('dummy_path.csv')
            
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (3, 3)
    
    def test_get_features_and_target(self):
        """Testa extração de features e target."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [1, 2, 1]
        })
        
        repo = CSVDataRepository()
        X, y = repo.get_features_and_target(data)
        
        assert X.shape == (3, 2)
        assert len(y) == 3
        assert y.name == 'target'
    
    def test_get_features_and_target_empty_dataframe(self):
        """Testa erro ao passar DataFrame vazio."""
        data = pd.DataFrame()
        repo = CSVDataRepository()
        
        with pytest.raises(ValueError):
            repo.get_features_and_target(data)


class TestPreprocessingService:
    """Testes do serviço de pré-processamento."""
    
    def test_normalize(self):
        """Testa normalização de dados."""
        X = pd.DataFrame({
            'feature1': [0, 10, 20],
            'feature2': [100, 200, 300]
        })
        
        service = SKLearnPreprocessingService()
        X_normalized, scaler = service.normalize(X)
        
        # Verificar se foi normalizado (média próxima a 0, std próximo a 1)
        assert abs(X_normalized.mean().mean()) < 0.1
        assert abs(X_normalized.std().mean() - 1) < 0.1
    
    def test_split_data_stratified(self):
        """Testa divisão estratificada de dados."""
        X = pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10, 20)
        })
        y = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 3])
        
        service = SKLearnPreprocessingService()
        X_train, X_test, y_train, y_test = service.split_data(
            X, y, test_size=0.3, stratify=True
        )
        
        assert len(X_train) == 7
        assert len(X_test) == 3
        assert len(y_train) == 7
        assert len(y_test) == 3
    
    def test_compute_class_weights(self):
        """Testa cálculo de pesos das classes."""
        y = pd.Series([1, 1, 1, 1, 2, 2, 3])  # Classes desbalanceadas
        
        service = SKLearnPreprocessingService()
        weights = service.compute_class_weights(y)
        
        assert len(weights) == 7
        assert all(w > 0 for w in weights)


class TestUseCases:
    """Testes dos casos de uso."""
    
    def test_load_data_use_case_success(self):
        """Testa load data use case com sucesso."""
        mock_repo = Mock()
        mock_df = pd.DataFrame({'a': [1, 2, 3]})
        mock_repo.load_data.return_value = mock_df
        
        use_case = LoadDataUseCase(mock_repo)
        request = LoadDataRequest(path='dummy.csv')
        response = use_case.execute(request)
        
        assert response.success is True
        assert response.data is not None
        assert response.error is None
    
    def test_load_data_use_case_error(self):
        """Testa load data use case com erro."""
        mock_repo = Mock()
        mock_repo.load_data.side_effect = FileNotFoundError("Arquivo não encontrado")
        
        use_case = LoadDataUseCase(mock_repo)
        request = LoadDataRequest(path='nonexistent.csv')
        response = use_case.execute(request)
        
        assert response.success is False
        assert response.error is not None


class TestIntegration:
    """Testes de integração entre componentes."""
    
    def test_full_pipeline_mock(self):
        """Testa pipeline completo com mocks."""
        # Criar dados mock
        mock_data = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100),
            'target': np.random.choice([1, 2, 3], 100)
        })
        
        # Usar repositório real
        repo = CSVDataRepository()
        X, y = repo.get_features_and_target(mock_data)
        
        # Usar serviço real
        preprocessing = SKLearnPreprocessingService()
        X_normalized, scaler = preprocessing.normalize(X)
        X_train, X_test, y_train, y_test = preprocessing.split_data(X_normalized, y)
        
        # Verificações
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
