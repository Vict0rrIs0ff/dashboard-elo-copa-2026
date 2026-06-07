import pandas as pd
import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")
st.title("Dashboard Elo Copa 2026")


@st.cache_data(show_spinner="Carregando o dataset...")
def carregar_dados():
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings",
        "elo_ratings_wc2026.csv",
    )


try:
    data = carregar_dados()
except Exception as erro:
    st.error("Não foi possível carregar o dataset diretamente do Kaggle.")
    st.exception(erro)
    st.stop()


c1, c2, c3 = st.columns(3)
c1.metric("Linhas", f"{len(data):,}".replace(",", "."))
c2.metric("Colunas", len(data.columns))
c3.metric("Seleções", data["country"].nunique())

st.subheader("Prévia do dataset bruto")
st.dataframe(data.head(20), use_container_width=True, hide_index=True)

st.subheader("Qualidade inicial")
quality = pd.DataFrame({
    "coluna": data.columns,
    "tipo": data.dtypes.astype(str).values,
    "valores_ausentes": data.isna().sum().values,
})

st.dataframe(quality, use_container_width=True, hide_index=True)
st.write(f"Duplicatas exatas encontradas: **{int(data.duplicated().sum())}**")