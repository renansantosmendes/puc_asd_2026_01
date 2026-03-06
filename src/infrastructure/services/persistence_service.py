"""
Infrastructure: Persistence Service

Responsável por salvar e carregar modelos.
"""

import pickle
from typing import Any
from src.domain.interfaces.repositories import ModelRepository


class PickleModelRepository(ModelRepository):
    """Implementação de ModelRepository usando pickle."""
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Salva um modelo em arquivo pickle.
        
        Args:
            model: Modelo para salvar
            filepath: Caminho do arquivo
            
        Raises:
            IOError: Se há erro ao salvar
        """
        try:
            pickle.dump(model, open(filepath, 'wb'))
        except IOError as e:
            raise IOError(f"Erro ao salvar modelo em {filepath}") from e
    
    def load_model(self, filepath: str) -> Any:
        """
        Carrega um modelo de arquivo pickle.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Modelo carregado
            
        Raises:
            FileNotFoundError: Se o arquivo não existe
            IOError: Se há erro ao carregar
        """
        try:
            model = pickle.load(open(filepath, 'rb'))
            return model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Modelo não encontrado em {filepath}") from e
        except IOError as e:
            raise IOError(f"Erro ao carregar modelo de {filepath}") from e
