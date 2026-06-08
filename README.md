# ⚽ Dashboard Streamlit Elo Copa 2026

Equipe: **Victor de Pinho Sampaio** e **Pedro Augusto Teixeira**.

A seção de análise exploratória apresenta indicadores e gráficos interativos. O notebook complementar registra a inspeção técnica.

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
- `notebooks/Projeto_Elo_Copa_2026_complementar.ipynb`
- `artifacts/analise_confederacoes.csv`
- `artifacts/top_15_selecoes_retrato_2026.csv`
