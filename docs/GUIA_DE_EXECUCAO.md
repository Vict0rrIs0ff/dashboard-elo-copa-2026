# Guia de execução do dashboard Streamlit

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline e testes

```powershell
python scripts/run_pipeline.py
pytest -q
```

## Dashboard

```powershell
streamlit run app.py
```

O aplicativo tenta obter o arquivo diretamente do Kaggle com `kagglehub`. Se isso não for possível, utiliza a cópia de contingência em `data/raw`.
