"""
Infrastructure: Model Training Service

Implementação concreta de ModelTrainingService.
Responsável por treinar diferentes tipos de modelos.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.domain.interfaces.repositories import ModelTrainingService


class ScikitLearnModelTrainingService(ModelTrainingService):
    """Implementação de ModelTrainingService usando scikit-learn."""
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Inicializa o serviço.
        
        Args:
            model_type: Tipo de modelo ('decision_tree', 'gradient_boosting')
        """
        self.model_type = model_type
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              sample_weights: Optional[np.ndarray] = None) -> Any:
        """
        Treina um modelo.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            sample_weights: Pesos para as amostras
            
        Returns:
            Modelo treinado
            
        Raises:
            ValueError: Se model_type não é válido
        """
        if self.model_type == 'decision_tree':
            return self._train_decision_tree(X_train, y_train)
        elif self.model_type == 'gradient_boosting':
            return self._train_gradient_boosting(X_train, y_train, sample_weights)
        else:
            raise ValueError(f"Tipo de modelo inválido: {self.model_type}")
    
    def _train_decision_tree(self, X_train: pd.DataFrame,
                            y_train: pd.Series) -> DecisionTreeClassifier:
        """Treina um Decision Tree."""
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def _train_gradient_boosting(self, X_train: pd.DataFrame,
                                y_train: pd.Series,
                                sample_weights: Optional[np.ndarray]) -> GradientBoostingClassifier:
        """Treina um Gradient Boosting."""
        model = GradientBoostingClassifier(
            max_depth=7,
            n_estimators=200,
            learning_rate=0.001,
            random_state=42
        )
        
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame,
                                y_train: pd.Series,
                                param_grid: Optional[Dict[str, list]] = None) -> Any:
        """
        Otimiza hiperparâmetros usando GridSearchCV.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            param_grid: Dicionário com hiperparâmetros para teste
            
        Returns:
            Objeto GridSearchCV com o melhor modelo
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01],
                'max_depth': [10, 15]
            }
        
        grid_search = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search
