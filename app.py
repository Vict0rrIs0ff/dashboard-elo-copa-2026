from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data, save_clean_data
from src.modeling import bayes_features, fit_model_bundle, predict_profile

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "elo_ratings_wc2026_clean.csv"
SUMMARY_PATH = ROOT / "artifacts" / "resumo_pipeline.json"

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Carregando o dataset diretamente do Kaggle...")
def load_dashboard_data() -> tuple[pd.DataFrame, dict]:
    raw = load_raw_data(RAW_PATH)
    cleaned, summary = clean_data(raw)
    save_clean_data(cleaned, CLEAN_PATH)
    return cleaned, summary


@st.cache_resource
def load_models(data: pd.DataFrame):
    return fit_model_bundle(data)


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def latest_team_profile(data: pd.DataFrame, country: str) -> pd.Series:
    return data.loc[data["country"] == country].sort_values("snapshot_date").iloc[-1]


def confusion_figure(matrix, title: str):
    return px.imshow(matrix, text_auto=True, x=["Predito: fora", "Predito: elite"], y=["Real: fora", "Real: elite"], labels={"x": "Classificação prevista", "y": "Classe real", "color": "Quantidade"}, title=title, aspect="auto")


data, summary = load_dashboard_data()
bundle = load_models(data)

st.title("⚽ Seleções de Elite na Copa do Mundo de 2026")
st.caption("Projeto de Estatística e Probabilidade com dados históricos de rating Elo")
st.info("A variável-alvo é **elite_top10**: vale 1 quando a seleção aparece entre as dez melhores posições do ranking Elo no retrato anual e 0 nos demais casos. Os modelos não utilizam o ranking atual nem o rating Elo atual como entradas.")

with st.expander("Como os dados foram tratados", expanded=False):
    st.write("Foram verificadas ausências, duplicatas, consistência das somas de partidas, tipos de dados e valores extremos. Os outliers plausíveis foram sinalizados e mantidos, pois representam trajetórias esportivas possíveis.")
    c1, c2, c3, c4 = st.columns(4)
    st.caption(f"Fonte carregada: **{summary['origem_dados']}**")
    c1.metric("Linhas originais", summary["linhas_iniciais"])
    c2.metric("Duplicatas semânticas removidas", summary["duplicatas_semanticas_removidas"])
    c3.metric("Retratos com menos de 30 jogos removidos", summary["registros_com_menos_de_30_partidas_removidos"])
    c4.metric("Linhas após tratamento", summary["linhas_finais"])

st.header("1. Análise exploratória dos dados")
latest_year = int(data["year"].max())
latest = data.loc[data["year"] == latest_year].sort_values(["rank", "country"]).copy()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Observações tratadas", f"{len(data):,}".replace(",", "."))
k2.metric("Seleções", data["country"].nunique())
k3.metric("Período", f"{int(data['year'].min())} a {latest_year}")
k4.metric("Observações de elite", percentage(data[TARGET_COLUMN].mean()))

left, right = st.columns(2)
with left:
    elite_label = data[TARGET_COLUMN].map({0: "Fora do Top 10", 1: "Elite (Top 10)"})
    fig = px.histogram(data.assign(classe=elite_label), x="rating_avg", color="classe", nbins=35, barmode="overlay", title="Distribuição do rating Elo médio histórico por classe", labels={"rating_avg": "Rating Elo médio histórico", "count": "Quantidade", "classe": "Classe"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Objetivo analítico: verificar se o histórico médio de força diferencia seleções de elite.")
with right:
    confederation = data.groupby("confederation", as_index=False).agg(observacoes=(TARGET_COLUMN, "size"), proporcao_elite=(TARGET_COLUMN, "mean")).sort_values("proporcao_elite", ascending=False)
    fig = px.bar(confederation, x="confederation", y="proporcao_elite", text_auto=".1%", title="Proporção de observações no Top 10 por confederação", labels={"confederation": "Confederação", "proporcao_elite": "Proporção de elite"})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Objetivo analítico: comparar a presença histórica no Top 10 entre confederações.")

left, right = st.columns(2)
with left:
    trend = data.groupby("year", as_index=False).agg(rating_mediano=("rating", "median"))
    fig = px.line(trend, x="year", y="rating_mediano", title="Evolução da mediana do rating Elo das seleções analisadas", labels={"year": "Ano", "rating_mediano": "Mediana do rating Elo"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Objetivo analítico: observar a evolução temporal do nível mediano das seleções qualificadas.")
with right:
    top_latest = latest.head(15).sort_values("rating")
    fig = px.bar(top_latest, x="rating", y="country", orientation="h", title=f"As 15 seleções com maior rating Elo no retrato de {latest_year}", labels={"rating": "Rating Elo atual", "country": "Seleção"}, text="rating")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Objetivo analítico: identificar as seleções mais fortes no retrato mais recente do dataset.")

correlation_columns = ["rating_avg", "win_rate", "draw_rate", "goals_for_per_match", "goals_against_per_match", "goal_diff_per_match", "matches_total", TARGET_COLUMN]
correlation = data[correlation_columns].corr()
fig = px.imshow(correlation, text_auto=".2f", title="Matriz de correlação entre indicadores selecionados", labels={"color": "Correlação"}, aspect="auto", zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)
st.caption("Objetivo analítico: identificar relações lineares que ajudam a interpretar a classificação.")

st.header("2. Avaliação temporal dos classificadores")
st.write(f"Os classificadores foram treinados com {bundle.train_rows} observações até 2018 e avaliados com {bundle.test_rows} observações de 2019 a {latest_year}. Para a simulação interativa, os modelos são reajustados com todo o conjunto tratado.")
metric_table = bundle.metrics.copy()
for column in ["acuracia", "precisao", "recall", "f1_score", "roc_auc"]:
    metric_table[column] = metric_table[column].map(lambda value: f"{value:.3f}")
st.dataframe(metric_table, use_container_width=True, hide_index=True)

matrix_columns = st.columns(3)
for column, (name, matrix) in zip(matrix_columns, bundle.confusion_matrices.items()):
    with column:
        st.plotly_chart(confusion_figure(matrix, name), use_container_width=True)

st.header("3. Classificação probabilística interativa")
st.write("Escolha uma seleção como ponto de partida e ajuste os atributos. A tela compara a probabilidade calculada pelo Teorema de Bayes com as estimativas dos dois algoritmos de classificação.")
countries = sorted(data["country"].unique().tolist())
base_country = st.selectbox("Carregar perfil histórico mais recente de uma seleção", countries, index=countries.index("Brazil") if "Brazil" in countries else 0)
base = latest_team_profile(data, base_country)

left, middle, right = st.columns(3)
with left:
    confederations = sorted(data["confederation"].unique().tolist())
    confederation = st.selectbox("Confederação", confederations, index=confederations.index(str(base["confederation"])))
    matches_total = st.number_input("Partidas acumuladas", min_value=30, max_value=int(data["matches_total"].max() * 1.2), value=int(base["matches_total"]), step=1)
    rating_avg = st.number_input("Rating Elo médio histórico", min_value=int(data["rating_avg"].min()), max_value=int(data["rating_avg"].max()), value=int(base["rating_avg"]), step=1)
with middle:
    win_rate = st.slider("Taxa de vitórias", 0.0, 1.0, float(base["win_rate"]), 0.01)
    draw_rate = st.slider("Taxa de empates", 0.0, 1.0, float(base["draw_rate"]), 0.01)
    neutral_share = st.slider("Proporção de jogos em campo neutro", 0.0, 1.0, float(base["neutral_share"]), 0.01)
with right:
    goals_for_per_match = st.number_input("Gols marcados por partida", min_value=0.0, max_value=float(data["goals_for_per_match"].max() * 1.2), value=float(base["goals_for_per_match"]), step=0.01, format="%.2f")
    goals_against_per_match = st.number_input("Gols sofridos por partida", min_value=0.0, max_value=float(data["goals_against_per_match"].max() * 1.2), value=float(base["goals_against_per_match"]), step=0.01, format="%.2f")
    is_host = st.checkbox("Seleção anfitriã", value=bool(base["is_host"]))

profile = pd.DataFrame([{"matches_total": matches_total, "win_rate": win_rate, "draw_rate": draw_rate, "goals_for_per_match": goals_for_per_match, "goals_against_per_match": goals_against_per_match, "goal_diff_per_match": goals_for_per_match - goals_against_per_match, "neutral_share": neutral_share, "rating_avg": rating_avg, "confederation": confederation, "is_host": int(is_host)}])
result = predict_profile(bundle, profile)
result["probabilidade_exibida"] = result["probabilidade_elite"].map(percentage)

cards = st.columns(3)
for card, row in zip(cards, result.to_dict(orient="records")):
    with card:
        st.metric(row["modelo"], row["probabilidade_exibida"])
        st.write(row["classificacao"])

fig = px.bar(result, x="modelo", y="probabilidade_elite", text_auto=".1%", title="Comparação das probabilidades de pertencer ao Top 10", labels={"modelo": "Método", "probabilidade_elite": "Probabilidade de elite"})
fig.update_yaxes(range=[0, 1], tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Ver as etapas do Teorema de Bayes para esta simulação"):
    bayes_model = bundle.live_models["Bayes manual"]
    details, posterior = bayes_model.explain_prediction(profile[bayes_features()])
    st.write("O método calcula a priori de cada classe, multiplica as verossimilhanças condicionais dos atributos observados e normaliza os resultados para obter as probabilidades a posteriori. Foi aplicada suavização de Laplace para evitar probabilidades iguais a zero.")
    display_details = details.copy()
    display_details["probabilidade"] = display_details["probabilidade"].map(lambda value: f"{value:.6f}")
    display_posterior = posterior.copy()
    display_posterior["probabilidade_posteriori"] = display_posterior["probabilidade_posteriori"].map(percentage)
    st.subheader("Probabilidades a priori e verossimilhanças")
    st.dataframe(display_details, use_container_width=True, hide_index=True)
    st.subheader("Probabilidades a posteriori")
    st.dataframe(display_posterior, use_container_width=True, hide_index=True)

st.divider()
st.caption("Fonte do dataset: Kaggle, 2026 FIFA World Cup — Historical Elo Ratings. Projeto acadêmico desenvolvido com apoio declarado de IA generativa e revisão da equipe.")
