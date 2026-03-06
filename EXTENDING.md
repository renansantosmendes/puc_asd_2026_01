# Guia de Extensão do Projeto

Este guia mostra como estender o projeto seguindo a arquitetura Clean Architecture e SOLID.

## 📋 Índice

1. [Adicionar um Novo Modelo](#adicionar-um-novo-modelo)
2. [Adicionar um Novo Serviço](#adicionar-um-novo-serviço)
3. [Adicionar um Novo Repositório](#adicionar-um-novo-repositório)
4. [Adicionar um Novo Use Case](#adicionar-um-novo-use-case)
5. [Trocar Implementação de um Serviço](#trocar-implementação-de-um-serviço)

---

## Adicionar um Novo Modelo

### Cenário: Adicionar suporte para XGBoost

### Passo 1: Instalar a Biblioteca

```bash
pip install xgboost
```

### Passo 2: Atualizar o Serviço de Treinamento

Edite `src/infrastructure/services/model_training_service.py`:

```python
from xgboost import XGBClassifier

class ScikitLearnModelTrainingService(ModelTrainingService):
    # ... código existente ...
    
    def train(self, X_train, y_train, sample_weights=None):
        if self.model_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, sample_weights)
        # ... outros modelos ...
    
    def _train_xgboost(self, X_train, y_train, sample_weights=None):
        """Treina um modelo XGBoost."""
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        return model
```

### Passo 3: Usar no Pipeline

Edite `src/presentation/cli.py`:

```python
# Treinar XGBoost
print("\n[4/6] Treinando XGBoost...")
train_response = self.train_model_use_case.execute(
    TrainModelRequest(
        model_type='xgboost',  # 👈 Novo tipo
        X_train=X_train,
        y_train=y_train,
        sample_weights=sample_weights
    )
)
```

---

## Adicionar um Novo Serviço

### Cenário: Adicionar Serviço de Tuning com Optuna

### Passo 1: Definir a Interface

Edite `src/domain/interfaces/repositories.py`:

```python
class HyperparameterTuningService(ABC):
    """Interface para tuning de hiperparâmetros."""
    
    @abstractmethod
    def optimize(self, X_train, y_train) -> Any:
        """Otimiza hiperparâmetros usando Optuna."""
        pass
```

### Passo 2: Implementar o Serviço

Crie `src/infrastructure/services/optuna_tuning_service.py`:

```python
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from src.domain.interfaces.repositories import HyperparameterTuningService

class OptunaHyperparameterService(HyperparameterTuningService):
    """Implementação de tuning com Optuna."""
    
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
    
    def optimize(self, X_train, y_train):
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            }
            
            model = GradientBoostingClassifier(**params, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_trial.params, study.best_trial.value
```

### Passo 3: Injetar na CLI

Edite `src/presentation/cli.py`:

```python
from src.infrastructure.services.optuna_tuning_service import OptunaHyperparameterService

class FetalHealthCLI:
    def __init__(self):
        # ... código existente ...
        self.optuna_service = OptunaHyperparameterService(n_trials=50)
```

---

## Adicionar um Novo Repositório

### Cenário: Adicionar Suporte para MongoDB

### Passo 1: Atualizar a Interface

Edite `src/domain/interfaces/repositories.py` (já existe, apenas exemplo de uso):

```python
class ModelRepository(ABC):
    """Interface para repositório de modelos."""
    # Interface já definida - usar para MongoDB
```

### Passo 2: Implementar o Repositório

Crie `src/infrastructure/repositories/mongo_model_repository.py`:

```python
from pymongo import MongoClient
import pickle
import base64
from src.domain.interfaces.repositories import ModelRepository

class MongoModelRepository(ModelRepository):
    """Implementação de ModelRepository com MongoDB."""
    
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client['fetal_health']
        self.collection = self.db['models']
    
    def save_model(self, model, filepath):
        """Salva modelo no MongoDB."""
        serialized = pickle.dumps(model)
        encoded = base64.b64encode(serialized).decode('utf-8')
        
        self.collection.insert_one({
            'name': filepath,
            'model': encoded,
            'timestamp': pd.Timestamp.now()
        })
    
    def load_model(self, filepath):
        """Carrega modelo do MongoDB."""
        doc = self.collection.find_one({'name': filepath})
        if not doc:
            raise FileNotFoundError(f"Modelo {filepath} não encontrado")
        
        decoded = base64.b64decode(doc['model'])
        return pickle.loads(decoded)
```

### Passo 3: Trocar a Injeção

Edite `src/presentation/cli.py`:

```python
from src.infrastructure.repositories.mongo_model_repository import MongoModelRepository

class FetalHealthCLI:
    def __init__(self):
        # ...
        self.persistence_service = MongoModelRepository()  # Trocar
```

---

## Adicionar um Novo Use Case

### Cenário: Adicionar Use Case para Predição em Batch

### Passo 1: Criar o Use Case

Crie `src/application/use_cases/batch_predict_use_case.py`:

```python
import pandas as pd
from typing import Optional
from src.application.dto.model_dto import BatchPredictRequest, BatchPredictResponse

class BatchPredictUseCase:
    """Use case para predição em batch."""
    
    def __init__(self, evaluation_service):
        self.evaluation_service = evaluation_service
    
    def execute(self, request: BatchPredictRequest) -> BatchPredictResponse:
        try:
            predictions = request.model.predict(request.X)
            
            return BatchPredictResponse(
                success=True,
                predictions=predictions,
                confidence_scores=request.model.predict_proba(request.X)
            )
        except Exception as e:
            return BatchPredictResponse(
                success=False,
                error=str(e)
            )
```

### Passo 2: Criar DTOs

Edite `src/application/dto/model_dto.py`:

```python
@dataclass
class BatchPredictRequest:
    """Request para predição em batch."""
    model: Any
    X: pd.DataFrame

@dataclass
class BatchPredictResponse:
    """Response para predição em batch."""
    success: bool
    predictions: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    error: Optional[str] = None
```

### Passo 3: Integrar na CLI

Edite `src/presentation/cli.py`:

```python
from src.application.use_cases.batch_predict_use_case import BatchPredictUseCase

class FetalHealthCLI:
    def __init__(self):
        # ...
        self.batch_predict_use_case = BatchPredictUseCase(self.evaluation_service)
    
    def predict_batch(self, model, X):
        """Realiza predição em batch."""
        response = self.batch_predict_use_case.execute(
            BatchPredictRequest(model=model, X=X)
        )
        return response
```

---

## Trocar Implementação de um Serviço

### Cenário: Trocar de Matplotlib para Plotly

### Passo 1: Criar Nova Implementação

Crie `src/infrastructure/services/plotly_visualization_service.py`:

```python
import plotly.graph_objects as go
from src.domain.interfaces.repositories import VisualizationService

class PlotlyVisualizationService(VisualizationService):
    """Implementação de VisualizationService usando Plotly."""
    
    def plot_class_distribution(self, data, title):
        """Plota distribuição usando Plotly."""
        fig = go.Figure(data=[
            go.Bar(x=data['fetal_health'].value_counts().index,
                   y=data['fetal_health'].value_counts().values)
        ])
        fig.update_layout(title=title)
        fig.show()
    
    # ... implementar outros métodos ...
```

### Passo 2: Injetar na CLI

Edite `src/presentation/cli.py`:

```python
from src.infrastructure.services.plotly_visualization_service import PlotlyVisualizationService

class FetalHealthCLI:
    def __init__(self):
        # ...
        # Trocar de Matplotlib para Plotly
        self.visualization_service = PlotlyVisualizationService()
```

---

## 🧪 Testando suas Extensões

Sempre adicione testes para novas funcionalidades:

```python
# tests/test_my_extension.py

import pytest
from src.infrastructure.services.xgboost_training import XGBoostTrainer

def test_xgboost_training():
    """Testa treinamento com XGBoost."""
    trainer = XGBoostTrainer()
    model = trainer.train(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')
```

Execute os testes:

```bash
pytest tests/test_my_extension.py -v
```

---

## ✅ Checklist para Extensões

- [ ] Interface criada/atualizada em `domain/interfaces/`
- [ ] Implementação criada em `infrastructure/`
- [ ] Testes criados em `tests/`
- [ ] Injeção de dependência atualizada
- [ ] Documentação atualizada
- [ ] Testes passando (`pytest tests/`)
- [ ] Sem violação de SOLID principles
- [ ] Sem quebra de código existente

---

## 📚 Recursos Adicionais

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Design Patterns](https://refactoring.guru/design-patterns)
