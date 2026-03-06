"""
Use Case: Train Model

Encapsula a lógica de negócio para treinamento de modelos.
"""

import time
import pandas as pd
import numpy as np
from src.domain.interfaces.repositories import ModelTrainingService
from src.application.dto.model_dto import TrainModelRequest, TrainModelResponse


class TrainModelUseCase:
    """Use case para treinamento de modelos."""
    
    def __init__(self, training_service: ModelTrainingService):
        """
        Inicializa o use case.
        
        Args:
            training_service: Serviço de treinamento
        """
        self.training_service = training_service
    
    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        """
        Executa o case de uso.
        
        Args:
            request: Request com dados e configurações
            
        Returns:
            TrainModelResponse com modelo treinado
        """
        try:
            start_time = time.time()
            
            # Treinar modelo
            model = self.training_service.train(
                request.X_train,
                request.y_train,
                sample_weights=request.sample_weights
            )
            
            training_time = time.time() - start_time
            
            return TrainModelResponse(
                success=True,
                model=model,
                training_time=training_time
            )
        except Exception as e:
            return TrainModelResponse(
                success=False,
                error=str(e)
            )
