from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.bayes_manual import ManualCategoricalNaiveBayes
from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")
st.title("⚽ Dashboard Elo Copa 2026")
st.caption("Análise exploratória e classificação probabilística com Bayes manual")

raw = load_raw_data(RAW_PATH)
data, summary = clean_data(raw)
st.caption(f"Fonte carregada: **{summary['origem_dados']}**")

st.header("1. Resumo da base tratada")
c1, c2, c3 = st.columns(3)
c1.metric("Observações", len(data))
c2.metric("Seleções", data["country"].nunique())
c3.metric("Elite histórica", f"{data[TARGET_COLUMN].mean() * 100:.1f}%")

fig = px.histogram(data, x="rating_avg", color=data[TARGET_COLUMN].map({0: "Fora do Top 10", 1: "Elite (Top 10)"}), nbins=35, title="Rating Elo médio histórico por classe")
st.plotly_chart(fig, use_container_width=True)

st.header("2. Simulação pelo Teorema de Bayes")
features = ["rating_avg", "win_rate", "goal_diff_per_match", "matches_total", "confederation"]
model = ManualCategoricalNaiveBayes(
    numeric_features=["rating_avg", "win_rate", "goal_diff_per_match", "matches_total"],
    categorical_features=["confederation"],
    alpha=1.0,
)
model.fit(data[features], data[TARGET_COLUMN])

left, right = st.columns(2)
with left:
    rating_avg = st.number_input("Rating Elo médio histórico", value=1750, step=10)
    win_rate = st.slider("Taxa de vitórias", 0.0, 1.0, 0.55, 0.01)
    goal_diff = st.number_input("Saldo de gols por partida", value=0.50, step=0.05)
with right:
    matches_total = st.number_input("Partidas acumuladas", min_value=30, value=350, step=10)
    confederation = st.selectbox("Confederação", sorted(data["confederation"].unique()))

profile = pd.DataFrame([{"rating_avg": rating_avg, "win_rate": win_rate, "goal_diff_per_match": goal_diff, "matches_total": matches_total, "confederation": confederation}])
probability = float(model.predict_proba(profile)[:, 1][0])
st.metric("Probabilidade posterior de pertencer ao Top 10", f"{probability * 100:.1f}%")

details, posterior = model.explain_prediction(profile)
with st.expander("Ver priori, verossimilhanças e posteriori"):
    st.dataframe(details, use_container_width=True, hide_index=True)
    st.dataframe(posterior, use_container_width=True, hide_index=True)
