"""
Use Case: Prepare Data

Encapsula a lógica de negócio para preparação de dados.
Normalização, divisão treino/teste e cálculo de pesos.
"""

import pandas as pd
import numpy as np
from src.domain.interfaces.repositories import PreprocessingService
from src.application.dto.model_dto import PrepareDataRequest, PrepareDataResponse


class PrepareDataUseCase:
    """Use case para preparação de dados."""
    
    def __init__(self, preprocessing_service: PreprocessingService):
        """
        Inicializa o use case.
        
        Args:
            preprocessing_service: Serviço de pré-processamento
        """
        self.preprocessing_service = preprocessing_service
    
    def execute(self, request: PrepareDataRequest) -> PrepareDataResponse:
        """
        Executa o case de uso.
        
        Args:
            request: Request com dados e parâmetros
            
        Returns:
            PrepareDataResponse com dados preparados
        """
        try:
            # Normalizar dados
            X_normalized, scaler = self.preprocessing_service.normalize(request.X)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = self.preprocessing_service.split_data(
                X_normalized,
                request.y,
                test_size=request.test_size,
                stratify=request.stratify
            )
            
            # Calcular pesos das classes
            sample_weights = self.preprocessing_service.compute_class_weights(y_train)
            
            return PrepareDataResponse(
                success=True,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                scaler=scaler,
                sample_weights=sample_weights
            )
        except Exception as e:
            return PrepareDataResponse(
                success=False,
                error=str(e)
            )
