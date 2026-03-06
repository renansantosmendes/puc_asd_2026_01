"""
Infrastructure: Preprocessing Service

Implementação concreta de PreprocessingService.
Responsável por normalização, divisão de dados e cálculo de pesos.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.domain.interfaces.repositories import PreprocessingService


class SKLearnPreprocessingService(PreprocessingService):
    """Implementação de PreprocessingService usando scikit-learn."""
    
    def normalize(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normaliza features usando StandardScaler.
        
        Args:
            X: DataFrame com features
            
        Returns:
            Tupla (X_normalized, scaler)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_normalized = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_normalized, scaler
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.3, stratify: bool = True) -> Tuple:
        """
        Divide dados em treino e teste.
        
        Args:
            X: Features
            y: Target
            test_size: Proporção para teste
            stratify: Se deve usar estratificação
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        stratify_arg = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_arg,
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def compute_class_weights(self, y: pd.Series) -> np.ndarray:
        """
        Calcula pesos para cada classe.
        
        Args:
            y: Series com as classes
            
        Returns:
            Array com pesos para cada amostra
        """
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )
        
        class_weights_dict = dict(zip(classes, weights))
        sample_weights = np.array([class_weights_dict[cls] for cls in y])
        
        return sample_weights
