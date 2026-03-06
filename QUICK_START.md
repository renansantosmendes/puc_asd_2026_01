# ⚡ Quick Start - API de Saúde Fetal

## 🚀 5 Passos para Começar

### 1️⃣ Clonar/Baixar o Projeto
```bash
git clone <repo-url>
cd puc_asd_2026_01
```

### 2️⃣ Configurar o Arquivo `.env`

Copie o `.env.example` para `.env`:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edite o arquivo `.env` e substitua os valores:

```env
MLFLOW_TRACKING_USERNAME=seu_usuario_dagshub
MLFLOW_TRACKING_PASSWORD=seu_token_dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/seu_usuario_dagshub/puc_asd_2026_01.mlflow
```

**Como obter o token:**
1. Acesse https://dagshub.com
2. Faça login com sua conta
3. Vá para **Settings > Access Tokens**
4. Copie o token (sem espaços)

### 3️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Verificar Configuração (Recomendado)

Antes de iniciar, execute o verificador:

```bash
python check_setup.py
```

Você deve ver:
```
✅ Python
✅ .env
✅ Variáveis
✅ Dependências
✅ MLflow
```

Se algo falhar, consulte: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### 5️⃣ Iniciar a API

```bash
python run_api.py
```

Você deve ver:
```
🚀 Iniciando API de Classificação de Saúde Fetal
✅ MLflow configurado com sucesso
✅ Modelo carregado com sucesso!
✅ API pronta para receber requisições!
```

---

## 🧪 Testando a API

Em outro terminal, execute o teste:

```bash
python example_api_test.py
```

---

## 📚 Próximos Passos

Após confirmar que a API está funcionando:

1. **Documentação Interativa:**
   - Swagger: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

2. **Leia os Guias:**
   - [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md) - Guia completo de uso
   - [api/README.md](api/README.md) - Documentação da API
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Solução de problemas

3. **Faça Predições:**
   - Use cURL, Python ou a documentação interativa
   - Consulte [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md) para exemplos

---

## ❌ Problema? Siga Este Checklist

- [ ] `.env` foi criado e configurado?
- [ ] Credenciais do DagsHub estão corretas?
- [ ] Token foi copiado **sem espaços em branco**?
- [ ] Dependências foram instaladas? (`pip install -r requirements.txt`)
- [ ] Você rodou `check_setup.py` e todos os testes passaram?
- [ ] Você deixou a API completar o startup (não pressionou Ctrl+C)?

Se ainda tiver problemas, consulte: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🎯 Comandos Úteis

### Iniciar a API em porta diferente
```bash
python run_api.py 8888
```

### Ver documentação interativa
```
http://localhost:8000/docs
```

### Fazer teste rápido
```bash
curl -X GET "http://localhost:8000/health"
```

### Limpar cache do MLflow (se necessário)
```bash
# Windows:
rmdir /s %USERPROFILE%\.mlflow

# Linux/Mac:
rm -rf ~/.mlflow
```

---

**Tempo estimado para configuração:** 5-10 minutos ⏱️

**Dúvidas?** Consulte a documentação completa ou [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
