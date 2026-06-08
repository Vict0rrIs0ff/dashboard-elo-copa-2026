from pathlib import Path

import plotly.express as px
import streamlit as st

from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data, save_clean_data

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "elo_ratings_wc2026_clean.csv"

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")
st.title("⚽ Dashboard Elo Copa 2026")
st.caption("Análise exploratória interativa")

raw = load_raw_data(RAW_PATH)
data, summary = clean_data(raw)
save_clean_data(data, CLEAN_PATH)
st.caption(f"Fonte carregada: **{summary['origem_dados']}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observações tratadas", summary["linhas_finais"])
c2.metric("Seleções", data["country"].nunique())
c3.metric("Período", f"{int(data['year'].min())} a {int(data['year'].max())}")
c4.metric("Observações de elite", f"{data[TARGET_COLUMN].mean() * 100:.1f}%")

with st.expander("Como os dados foram tratados"):
    st.json(summary)

left, right = st.columns(2)
with left:
    classe = data[TARGET_COLUMN].map({0: "Fora do Top 10", 1: "Elite (Top 10)"})
    fig = px.histogram(data.assign(classe=classe), x="rating_avg", color="classe", nbins=35, barmode="overlay", title="Distribuição do rating Elo médio por classe")
    st.plotly_chart(fig, use_container_width=True)
with right:
    conf = data.groupby("confederation", as_index=False).agg(proporcao_elite=(TARGET_COLUMN, "mean"))
    fig = px.bar(conf, x="confederation", y="proporcao_elite", text_auto=".1%", title="Proporção histórica de elite por confederação")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

trend = data.groupby("year", as_index=False).agg(rating_mediano=("rating", "median"))
fig = px.line(trend, x="year", y="rating_mediano", title="Evolução da mediana do rating Elo")
st.plotly_chart(fig, use_container_width=True)

latest_year = int(data["year"].max())
top = data.loc[data["year"] == latest_year].sort_values("rank").head(15).sort_values("rating")
fig = px.bar(top, x="rating", y="country", orientation="h", title=f"As 15 seleções com maior rating em {latest_year}")
st.plotly_chart(fig, use_container_width=True)
