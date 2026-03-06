"""
Use Case: Evaluate Model

Encapsula a lógica de negócio para avaliação de modelos.
"""

import pandas as pd
from src.domain.interfaces.repositories import EvaluationService
from src.application.dto.model_dto import EvaluateModelRequest, EvaluateModelResponse


class EvaluateModelUseCase:
    """Use case para avaliação de modelos."""
    
    def __init__(self, evaluation_service: EvaluationService):
        """
        Inicializa o use case.
        
        Args:
            evaluation_service: Serviço de avaliação
        """
        self.evaluation_service = evaluation_service
    
    def execute(self, request: EvaluateModelRequest) -> EvaluateModelResponse:
        """
        Executa o case de uso.
        
        Args:
            request: Request com modelo e dados de teste
            
        Returns:
            EvaluateModelResponse com resultados
        """
        try:
            results = self.evaluation_service.evaluate(
                request.model,
                request.X_test,
                request.y_test
            )
            
            return EvaluateModelResponse(
                success=True,
                accuracy=results['accuracy'],
                classification_report=results['classification_report'],
                predictions=results['predictions']
            )
        except Exception as e:
            return EvaluateModelResponse(
                success=False,
                error=str(e)
            )
