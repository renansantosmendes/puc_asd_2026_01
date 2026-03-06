# 🏥 Classificação de Saúde Fetal - Modelo de Machine Learning

**Autor**: Renan Santos Mendes  
**Email**: renansantosmendes@gmail.com

---

## ⚡ Quick Start (5 minutos)

**Quer começar rapidinho?** Leia o [QUICK_START.md](QUICK_START.md) para instruções passo a passo!

```bash
# 1. Configure o arquivo .env
cp .env.example .env
# → Edite o .env com suas credenciais DagsHub

# 2. Instale dependências
pip install -r requirements.txt

# 3. Verifique a configuração
python check_setup.py

# 4. Inicie a API
python run_api.py

# 5. Em outro terminal, teste
python example_api_test.py
```

**Documentação da API:** http://localhost:8000/docs

---

## 📊 Sobre o Projeto

Este projeto implementa um modelo de aprendizado de máquina para classificação de saúde fetal usando dados de Cardiotocografias (CTGs). O projeto segue **Clean Architecture** e os princípios **SOLID** para garantir código bem estruturado, testável e escalável.

### 🎯 Objetivo

Classificar a saúde fetal em três categorias:
- **Normal**: Saúde fetal normal
- **Suspeito**: Saúde fetal suspeita
- **Patológico**: Saúde fetal patológica

### 📈 Dataset

O dataset contém **2126 registros** de características extraídas de exames de Cardiotocografia, com **21 features** relacionadas a:
- Frequência cardíaca fetal (FCF)
- Movimentos fetais
- Contrações uterinas
- E muito mais

## 🏗️ Arquitetura

O projeto está organizado em camadas:

```
Domain Layer
    ↓ (depende de)
Application Layer
    ↓ (depende de)
Infrastructure Layer
    ↓ (depende de)
External Libraries
```

Para detalhes completos, veja [ARCHITECTURE.md](ARCHITECTURE.md).

## 📦 Estrutura de Arquivos

```
src/
├── domain/                 # Entidades e interfaces
├── application/            # Use cases e DTOs
├── infrastructure/         # Implementações de serviços
└── presentation/           # Interface CLI

main.py                     # Ponto de entrada
requirements.txt            # Dependências
environment.yml             # Ambiente Conda
```

## 🚀 Instalação e Execução

### Opção 1: Usar pip

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar o aplicativo
python main.py
```

### Opção 2: Usar Conda

```bash
# Criar ambiente
conda env create -f environment.yml

# Ativar ambiente
conda activate puc_asd_2026_01

# Executar o aplicativo
python main.py
```

## 📋 Pipeline do Aplicativo

O aplicativo executa automaticamente os seguintes passos:

1. **Carregamento de Dados**: Download do dataset CSV
2. **Visualização**: Distribuição das classes
3. **Preparação de Dados**: 
   - Normalização (StandardScaler)
   - Divisão treino/teste (70/30 com estratificação)
   - Cálculo de pesos para balanceamento de classes
4. **Treinamento**: Gradient Boosting Classifier
5. **Avaliação**: 
   - Acurácia
   - Classification Report
   - Matriz de Confusão
6. **Salvamento**: Modelo exportado em pickle

## 📊 Modelos Implementados

### Decision Tree
- Classifier simples e interpretável
- Bom baseline para comparação

### Gradient Boosting
- Modelo ensemble mais robusto
- Melhor desempenho em geral
- Suporta sample weights para balanceamento

## 🔧 Uso Avançado

### Integração com MLflow

Para usar MLflow (opcional):

```python
from src.infrastructure.services.mlflow_service import MLFlowRepository

mlflow_repo = MLFlowRepository()
mlflow_repo.setup()
mlflow_repo.log_metrics({'accuracy': 0.95})
```

### Adicionar Novo Modelo

1. Implementar em `infrastructure/services/model_training_service.py`
2. Atualizar `TrainModelRequest` se necessário
3. Usar em `presentation/cli.py`

### Trocar Implementação

Exemplo: usar MongoDB em vez de Pickle

```python
# Em presentation/cli.py
from src.infrastructure.services.mongo_persistence import MongoModelRepository

self.persistence_service = MongoModelRepository()
```

## 📈 Métricas de Desempenho

O modelo fornece:
- **Accuracy**: Proporção geral de predições corretas
- **Precision**: De todas as predições positivas, quantas foram corretas
- **Recall**: De todos os casos positivos, quantos foram detectados
- **F1-Score**: Média harmônica entre Precision e Recall

## 🧪 Testes

Para rodar testes (quando implementados):

```bash
pytest tests/
```

## 📚 Dependências Principais

- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Modelos de ML e pré-processamento
- **matplotlib/seaborn**: Visualizações
- **mlflow**: Rastreamento de experimentos (opcional)

## � Documentação Completa

| Documento | Descrição |
|-----------|-----------|
| [QUICK_START.md](QUICK_START.md) | ⚡ Guia rápido para começar (5 minutos) |
| [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md) | 📖 Guia completo de uso da API |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 🔧 Soluções para problemas comuns |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 🏗️ Detalhes da arquitetura Clean Architecture |
| [api/README.md](api/README.md) | 📡 Documentação específica da API |

---

## 🆘 Tendo Problemas?

### Antes de Começar
1. Execute `python check_setup.py` para verificar a configuração
2. Consulte [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para problemas comuns
3. Leia [QUICK_START.md](QUICK_START.md) para passo a passo detalhado

### Problema Comum: Erro de Modelo não Encontrado
```
⚠️  Aviso: Nenhum modelo encontrado no MLflow
```

**Solução:** Consulte a seção "Nenhum modelo encontrado no MLflow" em [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Problema Comum: Porta Já em Uso
```
OSError: [Errno 10048] Apenas um uso de cada endereço de socket...
```

**Solução:** Use outro `python run_api.py 8888`

---

## 🔐 Segurança

- ⚠️ **Nunca** comita o arquivo `.env` com credenciais reais
- ✅ Use `.env` apenas localmente para variáveis sensíveis
- ✅ Use `secrets` ou variáveis de ambiente em produção
- ✅ Veja `.gitignore` para arquivos excluídos

---

## 🎯 Fluxo de Trabalho Recomendado

```
1. Fazer Alterações
          ↓
2. Executar Testes (pytest)
          ↓
3. Fazer Predições com a API
          ↓
4. Registrar no MLflow/DagsHub
          ↓
5. Deploy
```

---

## 📞 Contato & Suporte

- **Autor**: Renan Santos Mendes
- **Email**: renansantosmendes@gmail.com
- **GitHub**: https://github.com/renansantosmendes

Para reportar bugs ou sugerir melhorias, consulte a documentação ou entre em contato.

---

## 📝 Licença

Este projeto é baseado em dados públicos e segue as licenças dos dados originais.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📞 Contato

- **Email**: renansantosmendes@gmail.com
- **GitHub**: [renansantosmendes](https://github.com/renansantosmendes)

## 📖 Referências

- [Clean Architecture - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**Última atualização**: Março de 2026
