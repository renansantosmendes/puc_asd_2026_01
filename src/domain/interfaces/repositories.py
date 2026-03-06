"""
Domain Interfaces - Define contratos que devem ser implementados.

Essas interfaces garantem que a lógica de negócio não depende
de implementações específicas (Dependency Inversion Principle).
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class DataRepository(ABC):
    """Interface para repositório de dados."""
    
    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        """Carrega dados de uma fonte."""
        pass
    
    @abstractmethod
    def get_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extrai features e target do dataset."""
        pass


class ModelRepository(ABC):
    """Interface para repositório de modelos."""
    
    @abstractmethod
    def save_model(self, model: Any, filepath: str) -> None:
        """Salva um modelo."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> Any:
        """Carrega um modelo."""
        pass


class MLFlowRepository(ABC):
    """Interface para integração com MLflow."""
    
    @abstractmethod
    def setup(self) -> bool:
        """Configura MLflow."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Registra métricas."""
        pass
    
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> None:
        """Registra um modelo."""
        pass


class PreprocessingService(ABC):
    """Interface para serviço de pré-processamento."""
    
    @abstractmethod
    def normalize(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Normaliza features."""
        pass
    
    @abstractmethod
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.3, stratify: bool = True) -> Tuple:
        """Divide dados em treino e teste."""
        pass
    
    @abstractmethod
    def compute_class_weights(self, y: pd.Series) -> np.ndarray:
        """Calcula pesos das classes."""
        pass


class ModelTrainingService(ABC):
    """Interface para serviço de treinamento de modelos."""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              sample_weights: np.ndarray = None) -> Any:
        """Treina um modelo."""
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, X_train: pd.DataFrame, 
                                 y_train: pd.Series) -> Any:
        """Otimiza hiperparâmetros."""
        pass


class EvaluationService(ABC):
    """Interface para serviço de avaliação."""
    
    @abstractmethod
    def evaluate(self, model: Any, X_test: pd.DataFrame, 
                 y_test: pd.Series) -> Dict[str, Any]:
        """Avalia um modelo."""
        pass


class VisualizationService(ABC):
    """Interface para serviço de visualização."""
    
    @abstractmethod
    def plot_class_distribution(self, data: pd.DataFrame, title: str) -> None:
        """Plota distribuição de classes."""
        pass
    
    @abstractmethod
    def plot_confusion_matrix(self, model: Any, X_test: pd.DataFrame, 
                             y_test: pd.Series, title: str) -> None:
        """Plota matriz de confusão."""
        pass
