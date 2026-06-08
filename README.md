# ⚽ Dashboard Streamlit Elo Copa 2026

Equipe: **Victor de Pinho Sampaio** e **Pedro Augusto Teixeira**.

Foram acrescentadas árvore de decisão, avaliação temporal, métricas e matrizes de confusão.

## Como executar o projeto

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute a aplicação

```bash
python scripts/run_pipeline.py
streamlit run app.py
```

## Fonte dos dados

O aplicativo tenta carregar o CSV diretamente do Kaggle com `kagglehub`. A pasta `data/raw` mantém uma cópia local apenas como contingência e rastreabilidade.

## Arquivos relevantes nesta etapa

- `app.py`
- `scripts/run_pipeline.py`
- `src/modeling.py`
- `artifacts/metricas_modelos.csv`
