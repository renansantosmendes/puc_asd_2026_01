# 🔧 Guia de Troubleshooting - API de Saúde Fetal

## ❌ Problemas Comuns e Soluções

---

### 1. ⚠️ "KeyboardInterrupt" durante o startup

**Sintoma:**
```
Application startup complete.
KeyboardInterrupt
asyncio.exceptions.CancelledError
```

**Causa:** O usuário pressionou Ctrl+C enquanto a API estava iniciando (especialmente durante o download do modelo).

**Solução:**
1. Deixe a API completar o startup (aguarde o log "✅ API pronta para receber requisições!")
2. Não interrompa com Ctrl+C durante a inicialização
3. Se precisar parar, aguarde a API estar completamente pronta

---

### 2. ❌ "Nenhum modelo encontrado no MLflow"

**Sintoma:**
```
⚠️  Aviso: Nenhum modelo encontrado no MLflow (experimento 0)
Certifique-se de que o modelo foi registrado corretamente
```

**Causas Possíveis:**
- URI do MLflow/DagsHub incorreta
- Credenciais inválidas
- Modelo não foi registrado no experimento 0
- Sem acesso à internet ou ao DagsHub

**Solução:**

1. **Verifique o arquivo `.env`:**
   ```env
   MLFLOW_TRACKING_USERNAME=renansantosmendes
   MLFLOW_TRACKING_PASSWORD=seu_token_aqui
   MLFLOW_TRACKING_URI=https://dagshub.com/renansantosmendes/puc_asd_2026_01.mlflow
   ```
   - ✅ A URI deve apontar para o **repositório correto** (puc_asd_2026_01)
   - ✅ A senha **não deve ter espaços em branco** no final
   - ✅ O username e repositório devem corresponder ao seu DagsHub

2. **Verifique as credenciais:**
   ```bash
   # No DagsHub, vá para Settings > Access tokens
   # Copie o token exato (sem espaços)
   ```

3. **Verifique a conexão com DagsHub:**
   ```bash
   # Teste se consegue acessar o MLflow
   # Abra no navegador: https://dagshub.com/seu_usuario/seu_repo/mlflow
   ```

4. **Verifique se o modelo foi registrado:**
   - Acesse o DagsHub
   - Vá para a aba MLflow do seu repositório
   - Verifique se há modelos no experimento com ID `0`

---

### 3. 🌐 "Erro ao buscar modelos: ConnectionError"

**Sintoma:**
```
⚠️  Erro ao buscar modelos: Unable to connect to MLflow
```

**Causas:**
- Sem conexão com a internet
- Firewall ou proxy bloqueando acesso ao DagsHub
- URI do MLflow incorreta

**Solução:**

1. **Teste a conexão com DagsHub:**
   ```bash
   ping dagshub.com
   # Ou tente acessar no navegador: https://dagshub.com
   ```

2. **Verifique se consegue fazer login no DagsHub:**
   - Abra: https://dagshub.com
   - Faça login com suas credenciais
   - Navegue até seu repositório

3. **Revise a URI no `.env`:**
   - Certifique-se de que é `https://` (não `http://`)
   - Verifique o formato: `https://dagshub.com/USERNAME/REPO.mlflow`

---

### 4. ❌ "Erro ao carregar o modelo"

**Sintoma:**
```
❌ Erro ao carregar o modelo: ...
O modelo pode estar corrompido ou a conexão falhou
```

**Causas:**
- Download interrompido
- Arquivo do modelo corrupto
- Versão incompatível do MLflow/sklearn

**Solução:**

1. **Reinicie a API:**
   ```bash
   # Pressione Ctrl+C completamente
   # Espere um momento
   python run_api.py
   ```

2. **Limpe o cache do MLflow:**
   ```bash
   # Remova a pasta de cache
   # Windows:
   rmdir /s %USERPROFILE%\.mlflow
   
   # Linux/Mac:
   rm -rf ~/.mlflow
   ```

3. **Reinstale as dependências:**
   ```bash
   pip install --upgrade mlflow scikit-learn
   ```

---

### 5. 🔴 "Porta 8000 já está em uso"

**Sintoma:**
```
OSError: [Errno 10048] Apenas um uso de cada endereço de socket é normalmente permitido
```

**Solução:**

**Opção 1:** Use uma porta diferente
```bash
python run_api.py 8888
```

**Opção 2:** Finalize o processo que está usando a porta
```bash
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

---

### 6. ⚠️ "Modelo não foi carregado" ao fazer predição

**Sintoma:**
```json
{
  "error": "❌ Modelo não foi carregado. Verifique as configurações do MLflow.",
  "prediction": null,
  "confidence": null,
  "probabilities": null
}
```

**Solução:**
1. Volte para o passo 2 (Nenhum modelo encontrado)
2. Aguarde o startup completar com "✅ Modelo carregado com sucesso!"
3. Verifique `/health` para confirmar que `model_loaded: true`

---

### 7. ❌ "Erro ao fazer predição: Não foi possível transformar os dados"

**Sintoma:**
```json
{
  "error": "❌ Erro ao fazer predição: X has ... features, but this StandardScaler is expecting 21 features"
}
```

**Causa:** Você enviou um número diferente de 21 features

**Solução:**
- Verifique que enviou **exatamente 21 valores** float
- Exemplo correto: `"features": [120, 0.5, ..., 0.0]` (21 valores)

---

### 8. 📊 Testando sem Erros

**Para confirmar que tudo está funcionando:**

```bash
# 1. Inicie a API
python run_api.py

# 2. Em outro terminal, teste
python example_api_test.py
```

**Resultado esperado:**
```
✅ Health Check realizado com sucesso!
   Status: ok
   Modelo carregado: ✅ Sim

📊 Testando: Caso Normal
   ✅ Predição realizada com sucesso!
   ...
```

---

### 9. 🆘 "Tudo estava funcionando, agora parou"

**Possíveis causas:**
- Arquivo `.env` foi apagado ou modificado
- Credenciais do DagsHub expiraram
- Modelo foi removido do repositório
- Mudança nas permissões de acesso

**Checklist de restauração:**
1. ✅ Arquivo `.env` existe e está correto?
2. ✅ Token do DagsHub ainda é válido?
3. ✅ Modelo ainda existe no repositório MLflow?
4. ✅ Conectividade com internet e DagsHub?
5. ✅ Versão do Python é 3.13+?
6. ✅ Dependências estão atualizadas? (`pip install -r requirements.txt`)

---

## 📝 Logs Úteis

### Para ver logs detalhados:

```bash
# Inicie com debug
uvicorn api.main:app --log-level debug
```

### Logs esperados no startup bem-sucedido:

```
======================================================================
🚀 Iniciando API de Classificação de Saúde Fetal
======================================================================
✅ MLflow configurado com sucesso
   URI: https://dagshub.com/...
   Usuário: ...
⏳ Buscando modelos registrados no MLflow...
✅ 1 modelo(s) encontrado(s)
⏳ Carregando modelo: ...
✅ Modelo carregado com sucesso!
✅ Scaler inicializado com sucesso
======================================================================
✅ API pronta para receber requisições!
======================================================================
```

---

## 🔍 Verificação de Saúde

Use o endpoint `/health` para verificar o status:

```bash
curl -X GET "http://localhost:8000/health"
```

**Resposta esperada:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

Se `model_loaded` for `false`, volte ao passo 2 desta guia.

---

## 📞 Obtendo Ajuda

Se nenhuma solução acima funcionou:

1. **Verifique a documentação completa:**
   - [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)
   - [api/README.md](api/README.md)

2. **Execute o teste de exemplo:**
   ```bash
   python example_api_test.py
   ```

3. **Contate o autor:**
   - Email: renansantosmendes@gmail.com
   - Forneça:
     - O erro completo (com logs)
     - Resultado do `/health`
     - Conteúdo do `.env` (sem a senha!)

---

**Última atualização:** Março 2026
