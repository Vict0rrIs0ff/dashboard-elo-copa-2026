from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data
from src.modeling import all_model_features, bayes_features, build_logistic_regression, build_manual_bayes

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")
st.title("⚽ Dashboard Elo Copa 2026")
st.caption("Comparação inicial: Bayes manual e regressão logística")

raw = load_raw_data(RAW_PATH)
data, summary = clean_data(raw)
st.caption(f"Fonte carregada: **{summary['origem_dados']}**")

st.header("1. Visão exploratória")
fig = px.histogram(data, x="rating_avg", color=data[TARGET_COLUMN].map({0: "Fora do Top 10", 1: "Elite (Top 10)"}), nbins=35, title="Rating Elo médio histórico por classe")
st.plotly_chart(fig, use_container_width=True)

bayes = build_manual_bayes()
logistic = build_logistic_regression()
bayes.fit(data[bayes_features()], data[TARGET_COLUMN])
logistic.fit(data[all_model_features()], data[TARGET_COLUMN])

st.header("2. Comparação interativa")
confederations = sorted(data["confederation"].unique())
left, right = st.columns(2)
with left:
    matches_total = st.number_input("Partidas acumuladas", min_value=30, value=350, step=10)
    rating_avg = st.number_input("Rating Elo médio histórico", value=1750, step=10)
    win_rate = st.slider("Taxa de vitórias", 0.0, 1.0, 0.55, 0.01)
    draw_rate = st.slider("Taxa de empates", 0.0, 1.0, 0.22, 0.01)
with right:
    goals_for = st.number_input("Gols marcados por partida", value=1.55, step=0.05)
    goals_against = st.number_input("Gols sofridos por partida", value=1.10, step=0.05)
    neutral_share = st.slider("Proporção em campo neutro", 0.0, 1.0, 0.12, 0.01)
    confederation = st.selectbox("Confederação", confederations)
    is_host = st.checkbox("Seleção anfitriã")

profile = pd.DataFrame([{"matches_total": matches_total, "win_rate": win_rate, "draw_rate": draw_rate, "goals_for_per_match": goals_for, "goals_against_per_match": goals_against, "goal_diff_per_match": goals_for - goals_against, "neutral_share": neutral_share, "rating_avg": rating_avg, "confederation": confederation, "is_host": int(is_host)}])
results = pd.DataFrame([
    {"modelo": "Bayes manual", "probabilidade_elite": float(bayes.predict_proba(profile[bayes_features()])[:, 1][0])},
    {"modelo": "Regressão logística", "probabilidade_elite": float(logistic.predict_proba(profile[all_model_features()])[:, 1][0])},
])
fig = px.bar(results, x="modelo", y="probabilidade_elite", text_auto=".1%", title="Probabilidade estimada de pertencer ao Top 10")
fig.update_yaxes(range=[0, 1], tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)
