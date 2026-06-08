$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest -q
streamlit run app.py
