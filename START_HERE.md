# Comece por aqui

O produto principal é o dashboard em Streamlit.

## Execução rápida no Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest -q
streamlit run app.py
```

O carregamento principal do dataset é realizado diretamente pelo KaggleHub. A cópia existente em `data/raw` é usada somente se o acesso online não estiver disponível.

## Arquivos centrais

- `app.py`: dashboard principal;
- `src/data_processing.py`: carregamento, limpeza e engenharia de atributos;
- `src/bayes_manual.py`: implementação manual do Teorema de Bayes;
- `src/modeling.py`: regressão logística, árvore de decisão e comparação;
- `notebooks/Projeto_Elo_Copa_2026_complementar.ipynb`: apoio técnico;
- `reports/RELATORIO_TECNICO.docx`: relatório final.
