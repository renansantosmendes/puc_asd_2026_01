"""
Infrastructure: Visualization Service

Implementação concreta de VisualizationService.
Responsável por gerar visualizações e gráficos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.metrics import ConfusionMatrixDisplay
from src.domain.interfaces.repositories import VisualizationService


class MatplotlibVisualizationService(VisualizationService):
    """Implementação de VisualizationService usando matplotlib e seaborn."""
    
    def plot_class_distribution(self, data: pd.DataFrame, title: str) -> None:
        """
        Plota distribuição de classes.
        
        Args:
            data: DataFrame com coluna 'fetal_health'
            title: Título do gráfico
        """
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(
            x='fetal_health', data=data,
            palette='viridis', hue='fetal_health',
            legend=False
        )
        
        plt.title(title)
        plt.xlabel('Saúde Fetal (1: Normal, 2: Suspeito, 3: Patológico)')
        plt.ylabel('Quantidade')
        
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10), textcoords='offset points'
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model: Any, X_test: pd.DataFrame,
                             y_test: pd.Series, title: str) -> None:
        """
        Plota matriz de confusão.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            title: Título do gráfico
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=['Normal', 'Suspeito', 'Patológico'],
            ax=ax, cmap='Blues'
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()
