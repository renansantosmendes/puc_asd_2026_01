"""
Main Entry Point

Ponto de entrada do aplicativo.
Executa o pipeline de classificação de saúde fetal.
"""

from src.presentation.cli import FetalHealthCLI


def main():
    """Função principal."""
    # URL do dataset
    data_path = 'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv'
    
    # Criar instância da CLI
    cli = FetalHealthCLI()
    
    # Executar pipeline
    cli.run(data_path)


if __name__ == "__main__":
    main()
