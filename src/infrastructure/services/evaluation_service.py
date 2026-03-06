"""
Infrastructure: Evaluation Service

Implementação concreta de EvaluationService.
Responsável por avaliar modelos e gerar métricas.
"""

import pandas as pd
from typing import Any, Dict
from sklearn.metrics import accuracy_score, classification_report
from src.domain.interfaces.repositories import EvaluationService


class MetricsEvaluationService(EvaluationService):
    """Implementação de EvaluationService."""
    
    def evaluate(self, model: Any, X_test: pd.DataFrame,
                y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia um modelo.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com resultados (accuracy, classification_report, predictions)
        """
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test, predictions,
            target_names=['Normal', 'Suspeito', 'Patológico']
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }
