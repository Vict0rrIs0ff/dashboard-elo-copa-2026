# ⚽ Dashboard Streamlit Elo Copa 2026

Equipe: **Victor de Pinho Sampaio** e **Pedro Augusto Teixeira**.

Foram acrescentados relatórios em Markdown, roteiro de apresentação e documentação acadêmica.

## Como executar o projeto

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute a aplicação

```bash
python scripts/run_pipeline.py
pytest -q
streamlit run app.py
```

## Fonte dos dados

O aplicativo tenta carregar o CSV diretamente do Kaggle com `kagglehub`. A pasta `data/raw` mantém uma cópia local apenas como contingência e rastreabilidade.

## Arquivos relevantes nesta etapa

- `app.py`
- `reports/RELATORIO_TECNICO.md`
- `reports/RELATORIO_EXECUTIVO.md`
- `docs/ROTEIRO_DE_DEFESA.md`
- `docs/PERGUNTAS_ARGUICAO.md`
