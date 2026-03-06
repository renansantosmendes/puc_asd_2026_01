"""
Domain Entities - Representa as entidades principais do domínio.

Esse módulo contém as classes que representam os conceitos principais
do negócio (fetal health classification).
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Tuple, Optional


@dataclass
class FetalHealthDataset:
    """
    Entidade que representa o dataset de saúde fetal.
    
    Attributes:
        X: Features do dataset
        y: Target (saúde fetal)
        feature_names: Nomes das features
    """
    X: pd.DataFrame
    y: pd.Series
    feature_names: list


@dataclass
class TrainTestSplit:
    """
    Entidade que representa a divisão treino/teste.
    
    Attributes:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class ModelEvaluationResult:
    """
    Entidade que representa o resultado da avaliação de um modelo.
    
    Attributes:
        accuracy: Acurácia do modelo
        classification_report: Relatório de classificação
        predictions: Predições do modelo
    """
    accuracy: float
    classification_report: str
    predictions: np.ndarray
