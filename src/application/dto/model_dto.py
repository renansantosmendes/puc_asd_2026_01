"""
Application DTOs - Data Transfer Objects para transferência entre camadas.

DTOs encapsulam dados trafegando entre a camada de aplicação
e apresentação, desacoplando as camadas.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class LoadDataRequest:
    """Request para carregamento de dados."""
    path: str


@dataclass
class LoadDataResponse:
    """Response para carregamento de dados."""
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    message: Optional[str] = None


@dataclass
class PrepareDataRequest:
    """Request para preparação de dados."""
    X: pd.DataFrame
    y: pd.Series
    test_size: float = 0.3
    stratify: bool = True
    random_state: int = 42


@dataclass
class PrepareDataResponse:
    """Response para preparação de dados."""
    success: bool
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    scaler: Optional[Any] = None
    sample_weights: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class TrainModelRequest:
    """Request para treinamento de modelo."""
    model_type: str  # 'decision_tree', 'gradient_boosting'
    X_train: pd.DataFrame
    y_train: pd.Series
    sample_weights: Optional[np.ndarray] = None
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class TrainModelResponse:
    """Response para treinamento de modelo."""
    success: bool
    model: Optional[Any] = None
    training_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class EvaluateModelRequest:
    """Request para avaliação de modelo."""
    model: Any
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class EvaluateModelResponse:
    """Response para avaliação de modelo."""
    success: bool
    accuracy: Optional[float] = None
    classification_report: Optional[str] = None
    predictions: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class SaveModelRequest:
    """Request para salvamento de modelo."""
    model: Any
    filepath: str


@dataclass
class SaveModelResponse:
    """Response para salvamento de modelo."""
    success: bool
    filepath: Optional[str] = None
    error: Optional[str] = None


@dataclass
class HyperparameterOptimizationRequest:
    """Request para otimização de hiperparâmetros."""
    X_train: pd.DataFrame
    y_train: pd.Series
    param_grid: Dict[str, list]
    cv: int = 3


@dataclass
class HyperparameterOptimizationResponse:
    """Response para otimização de hiperparâmetros."""
    success: bool
    best_model: Optional[Any] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    optimization_time: Optional[float] = None
    error: Optional[str] = None
