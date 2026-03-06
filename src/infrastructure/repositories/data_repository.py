"""
Infrastructure: Data Repository

Implementação concreta de DataRepository.
Responsável por carregar e processar dados do CSV.
"""

import pandas as pd
from typing import Tuple
from src.domain.interfaces.repositories import DataRepository


class CSVDataRepository(DataRepository):
    """Implementação de DataRepository para arquivos CSV."""
    
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            path: Caminho para o arquivo CSV (URL ou arquivo local)
            
        Returns:
            DataFrame com os dados carregados
            
        Raises:
            FileNotFoundError: Se o arquivo não existe
            pd.errors.ParserError: Se há erro ao parsear o CSV
        """
        try:
            data = pd.read_csv(path)
            return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Arquivo não encontrado: {path}") from e
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Erro ao parsear CSV: {path}") from e
    
    def get_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extrai features e target do dataset.
        
        Assume que a última coluna é o target.
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            Tupla (X, y) com features e target
            
        Raises:
            ValueError: Se o dataset está vazio
        """
        if data.empty:
            raise ValueError("Dataset está vazio")
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        return X, y
