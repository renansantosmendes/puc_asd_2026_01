# Arquitetura do Projeto - Classificação de Saúde Fetal

## 📋 Overview

Este projeto segue os princípios de **Clean Architecture** e **SOLID**, organizando o código em camadas bem definidas e independentes.

## 🏗️ Estrutura de Pastas

```
project/
├── src/
│   ├── domain/                    # Camada de Domínio (Entidades e Interfaces)
│   │   ├── entities/              # Entidades principais do negócio
│   │   │   └── model.py
│   │   └── interfaces/            # Contratos e abstrações
│   │       └── repositories.py
│   │
│   ├── application/               # Camada de Aplicação (Use Cases e DTOs)
│   │   ├── use_cases/             # Casos de uso da aplicação
│   │   │   ├── load_data_use_case.py
│   │   │   ├── prepare_data_use_case.py
│   │   │   ├── train_model_use_case.py
│   │   │   └── evaluate_model_use_case.py
│   │   └── dto/                   # Data Transfer Objects
│   │       └── model_dto.py
│   │
│   ├── infrastructure/            # Camada de Infraestrutura (Implementações)
│   │   ├── repositories/          # Implementações de repositórios
│   │   │   └── data_repository.py
│   │   ├── services/              # Implementações de serviços
│   │   │   ├── preprocessing_service.py
│   │   │   ├── model_training_service.py
│   │   │   ├── evaluation_service.py
│   │   │   ├── visualization_service.py
│   │   │   └── persistence_service.py
│   │   └── config/                # Configurações
│   │       └── __init__.py
│   │
│   └── presentation/              # Camada de Apresentação (Interface com usuário)
│       └── cli.py
│
├── main.py                        # Ponto de entrada do aplicativo
├── requirements.txt               # Dependências Python
├── environment.yml                # Ambiente Conda
└── ARCHITECTURE.md               # Este arquivo
```

## 🎯 Camadas da Arquitetura

### 1. **Domain Layer** (Camada de Domínio)

Contém as entidades principais e interfaces (abstrações) do domínio de negócio.

**Responsabilidades:**
- Definir entidades que representam conceitos do domínio (ex: `FetalHealthDataset`)
- Definir interfaces que especificam contratos que devem ser implementados
- Manter a lógica independente de qualquer framework

**Arquivos principais:**
- `entities/model.py`: Entidades de domínio
- `interfaces/repositories.py`: Interfaces abstratas

**SOLID Principles aplicados:**
- ✅ Interface Segregation: Interfaces específicas e bem definidas
- ✅ Dependency Inversion: Código de domínio não depende de implementações

### 2. **Application Layer** (Camada de Aplicação)

Contém os casos de uso (use cases) que implementam a lógica de negócio e DTOs para transferência de dados.

**Responsabilidades:**
- Orquestrar o fluxo de negócio
- Coordenar repositórios e serviços
- Transformar dados entre entidades e DTOs
- Capturar erros e retornar respostas padronizadas

**Arquivos principais:**
- `use_cases/`: Implementações dos casos de uso
- `dto/model_dto.py`: Objetos de transferência de dados

**SOLID Principles aplicados:**
- ✅ Single Responsibility: Cada use case tem uma única responsabilidade
- ✅ Open/Closed: Aberto para extensão, fechado para modificação
- ✅ Dependency Inversion: Depende de abstrações do domain layer

### 3. **Infrastructure Layer** (Camada de Infraestrutura)

Contém implementações concretas de serviços e repositórios usando bibliotecas específicas.

**Responsabilidades:**
- Implementar interfaces do domain layer
- Encapsular detalhes de bibliotecas externas (scikit-learn, pandas, etc)
- Fornecer acesso a recursos (BD, arquivos, APIs)

**Arquivos principais:**
- `repositories/data_repository.py`: Carregamento de dados (CSV)
- `services/preprocessing_service.py`: Pré-processamento (scikit-learn)
- `services/model_training_service.py`: Treinamento de modelos
- `services/evaluation_service.py`: Avaliação de modelos
- `services/visualization_service.py`: Visualizações (matplotlib, seaborn)
- `services/persistence_service.py`: Salvamento de modelos (pickle)

**SOLID Principles aplicados:**
- ✅ Dependency Inversion: Implementa interfaces do domain layer
- ✅ Single Responsibility: Cada serviço tem uma função específica
- ✅ Liskov Substitution: Pode ser substituído por outra implementação

### 4. **Presentation Layer** (Camada de Apresentação)

Interface com o usuário. Coordena use cases e exibe resultados.

**Responsabilidades:**
- Orquestrar os use cases em uma sequência lógica
- Exibir resultados e feedback ao usuário
- Traduzir entrada do usuário em requests

**Arquivos principais:**
- `cli.py`: Interface de linha de comando

## 🔄 Fluxo de Dados

```
User Input
    ↓
Presentation Layer (CLI)
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Entities & Interfaces)
    ↓
Infrastructure Layer (Services & Repositories)
    ↓
External Libraries (scikit-learn, pandas, etc)
```

## 📐 SOLID Principles

### 1. **Single Responsibility Principle (SRP)**
- Cada classe tem apenas uma razão para mudar
- `PreprocessingService`: apenas pré-processamento
- `TrainModelUseCase`: apenas treinamento

### 2. **Open/Closed Principle (OCP)**
- Aberto para extensão, fechado para modificação
- Novos serviços podem ser adicionados sem modificar código existente

### 3. **Liskov Substitution Principle (LSP)**
- Subclasses podem ser substituídas por suas superclasses
- `PickleModelRepository` pode ser substituído por `MongoModelRepository`

### 4. **Interface Segregation Principle (ISP)**
- Interfaces específicas em vez de genéricas
- `PreprocessingService` não depende de métodos de treinamento

### 5. **Dependency Inversion Principle (DIP)**
- Depender de abstrações, não de implementações concretas
- `FetalHealthCLI` depende de interfaces, não de classes concretas

## 🚀 Como Executar

### Instalação

```bash
# Usar arquivo requirements.txt
pip install -r requirements.txt

# OU usar conda com environment.yml
conda env create -f environment.yml
conda activate puc_asd_2026_01
```

### Executar o Aplicativo

```bash
python main.py
```

## 🔧 Adicionando Novos Modelos

### Passo 1: Adicionar Tipo de Modelo
```python
# Em infrastructure/services/model_training_service.py
elif self.model_type == 'xgboost':
    return self._train_xgboost(X_train, y_train, sample_weights)

def _train_xgboost(self, ...):
    # Implementação
```

### Passo 2: Usar no Use Case
```python
train_response = self.train_model_use_case.execute(
    TrainModelRequest(
        model_type='xgboost',
        ...
    )
)
```

## 🔄 Substituindo Implementações

Exemplo: Trocar de Pickle para MongoDB

```python
# 1. Criar nova implementação
class MongoModelRepository(ModelRepository):
    def save_model(self, model, filepath):
        # Implementação com MongoDB
        pass

# 2. Injetar na CLI
class FetalHealthCLI:
    def __init__(self):
        self.persistence_service = MongoModelRepository()
```

## 📚 Benefícios da Arquitetura

✅ **Testabilidade**: Cada camada pode ser testada independentemente  
✅ **Manutenibilidade**: Código bem organizado e fácil de entender  
✅ **Escalabilidade**: Fácil adicionar novas features e modelos  
✅ **Reusabilidade**: Componentes podem ser reutilizados  
✅ **Flexibilidade**: Trocar implementações sem afetar o resto do código  

## 📝 Licença

Este projeto segue a licença do repositório original.
