from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import TARGET_COLUMN, clean_data, load_raw_data, save_clean_data
from src.modeling import fit_model_bundle

RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "elo_ratings_wc2026_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def safe_name(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii").replace(" ", "_")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(RAW_PATH)
    cleaned, cleaning_summary = clean_data(raw)
    save_clean_data(cleaned, CLEAN_PATH)
    bundle = fit_model_bundle(cleaned)
    bundle.metrics.to_csv(ARTIFACTS_DIR / "metricas_modelos.csv", index=False)
    for name, matrix in bundle.confusion_matrices.items():
        pd.DataFrame(matrix, index=["real_fora_top10", "real_elite_top10"], columns=["predito_fora_top10", "predito_elite_top10"]).to_csv(ARTIFACTS_DIR / f"matriz_confusao_{safe_name(name)}.csv")
    cleaned.loc[cleaned["year"] == cleaned["year"].max(), ["country", "rank", "rating", "rating_avg", "confederation", TARGET_COLUMN]].sort_values("rank").head(15).to_csv(ARTIFACTS_DIR / "top_15_selecoes_retrato_2026.csv", index=False)
    cleaned.groupby("confederation").agg(observacoes=(TARGET_COLUMN, "size"), observacoes_elite=(TARGET_COLUMN, "sum"), proporcao_elite=(TARGET_COLUMN, "mean"), rating_medio=("rating_avg", "mean")).sort_values("proporcao_elite", ascending=False).to_csv(ARTIFACTS_DIR / "analise_confederacoes.csv")
    summary = {**cleaning_summary, "avaliacao_temporal": {"treino_ate": 2018, "linhas_treino": bundle.train_rows, "teste_de": 2019, "teste_ate": int(cleaned["year"].max()), "linhas_teste": bundle.test_rows}}
    (ARTIFACTS_DIR / "resumo_pipeline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Pipeline concluído.")
    print(f"Dados tratados: {CLEAN_PATH}")
    print(f"Linhas finais: {len(cleaned)}")
    print(bundle.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
