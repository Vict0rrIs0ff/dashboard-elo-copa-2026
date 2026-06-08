from pathlib import Path

import streamlit as st

from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data, save_clean_data

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "elo_ratings_wc2026_clean.csv"

st.set_page_config(page_title="Elo Copa 2026", page_icon="⚽", layout="wide")
st.title("⚽ Dashboard Elo Copa 2026")
st.caption("Tratamento e limpeza dos dados")

raw = load_raw_data(RAW_PATH)
data, summary = clean_data(raw)
save_clean_data(data, CLEAN_PATH)
st.caption(f"Fonte carregada: **{summary['origem_dados']}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Linhas originais", summary["linhas_iniciais"])
c2.metric("Duplicatas semânticas removidas", summary["duplicatas_semanticas_removidas"])
c3.metric("Retratos curtos removidos", summary["registros_com_menos_de_30_partidas_removidos"])
c4.metric("Linhas tratadas", summary["linhas_finais"])

st.subheader("Decisões de tratamento")
st.write("Foram validadas datas, somas de partidas, resultados, ranking e gols. Também foram removidas duplicatas semânticas e observações com menos de 30 partidas acumuladas.")

st.subheader("Prévia da base tratada")
st.dataframe(data.head(20), use_container_width=True, hide_index=True)
st.download_button("Baixar CSV tratado", data.to_csv(index=False).encode("utf-8"), "elo_ratings_wc2026_clean.csv", "text/csv")

st.subheader("Distribuição inicial da variável-alvo")
target = data[TARGET_COLUMN].map({0: "Fora do Top 10", 1: "Elite (Top 10)"}).value_counts().rename_axis("classe").to_frame("observacoes")
st.dataframe(target, use_container_width=True)
