# ⚽ Dashboard Streamlit Elo Copa 2026

Equipe: **Victor de Pinho Sampaio** e **Pedro Augusto Teixeira**.

A aplicação foi desenvolvida em Streamlit para analisar ratings Elo históricos das seleções classificadas para a Copa do Mundo de 2026. O dashboard reúne tratamento dos dados, análise exploratória, Teorema de Bayes implementado manualmente, regressão logística, árvore de decisão e comparação visual dos classificadores.

## Como executar o projeto

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute o pipeline

```bash
python scripts/run_pipeline.py
```

### 3. Inicie o dashboard

```bash
streamlit run app.py
```

## Fonte dos dados

O aplicativo tenta carregar o CSV diretamente do Kaggle por meio da biblioteca `kagglehub`.

Dataset: **2026 FIFA World Cup Historical Elo Ratings**  
Fonte: https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings

A pasta `data/raw` mantém uma cópia local do arquivo somente como contingência e para preservar a rastreabilidade da base utilizada.

## Arquivos relevantes nesta etapa

- `app.py`
- `src/data_processing.py`
- `src/bayes_manual.py`
- `src/modeling.py`
- `scripts/run_pipeline.py`
- `notebooks/Projeto_Elo_Copa_2026_complementar.ipynb`
- `reports/RELATORIO_TECNICO.docx`
- `requirements.txt`

## Variável-alvo

A variável `elite_top10` identifica se a seleção aparece entre as dez primeiras posições do ranking Elo no retrato anual:

- `1`: seleção de elite, presente no Top 10;
- `0`: seleção fora do Top 10.

## Métodos comparados

- Teorema de Bayes implementado manualmente;
- regressão logística;
- árvore de decisão.

## Observação

Os modelos não utilizam diretamente o ranking Elo atual nem o rating Elo atual como variáveis de entrada, evitando vazamento de informação.
