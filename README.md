# ⚽ Dashboard Streamlit Elo Copa 2026

Equipe: **Victor de Pinho Sampaio** e **Pedro Augusto Teixeira**.

O dashboard executa a limpeza, apresenta o impacto dos tratamentos e oferece o download da base tratada.

## Como executar o projeto

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute a aplicação

```bash
python scripts/run_cleaning.py
streamlit run app.py
```

## Fonte dos dados

O aplicativo tenta carregar o CSV diretamente do Kaggle com `kagglehub`. A pasta `data/raw` mantém uma cópia local apenas como contingência e rastreabilidade.

## Arquivos relevantes nesta etapa

- `app.py`
- `src/data_processing.py`
- `scripts/run_cleaning.py`
- `data/processed/elo_ratings_wc2026_clean.csv`
- `artifacts/resumo_limpeza.json`
