"""
Use Case: Load Data

Encapsula a lógica de negócio para carregamento de dados.
Segue o princípio de Single Responsibility.
"""

from typing import Optional
import pandas as pd
from src.domain.interfaces.repositories import DataRepository
from src.application.dto.model_dto import LoadDataRequest, LoadDataResponse


class LoadDataUseCase:
    """Use case para carregamento de dados."""
    
    def __init__(self, data_repository: DataRepository):
        """
        Inicializa o use case.
        
        Args:
            data_repository: Implementação de DataRepository
        """
        self.data_repository = data_repository
    
    def execute(self, request: LoadDataRequest) -> LoadDataResponse:
        """
        Executa o case de uso.
        
        Args:
            request: Request com o caminho dos dados
            
        Returns:
            LoadDataResponse com o resultado
        """
        try:
            data = self.data_repository.load_data(request.path)
            
            return LoadDataResponse(
                success=True,
                data=data,
                message=f"Dados carregados com sucesso: {data.shape[0]} linhas, {data.shape[1]} colunas"
            )
        except Exception as e:
            return LoadDataResponse(
                success=False,
                error=str(e)
            )
