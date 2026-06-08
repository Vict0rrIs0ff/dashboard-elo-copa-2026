from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import pandas as pd

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    kagglehub = None
    KaggleDatasetAdapter = None

MINIMUM_MATCHES = 30
TARGET_COLUMN = "elite_top10"
KAGGLE_DATASET = "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings"
KAGGLE_FILE = "elo_ratings_wc2026.csv"

RAW_COLUMNS = [
    "year",
    "snapshot_date",
    "country",
    "rank",
    "country_code",
    "rating",
    "rank_max",
    "rating_max",
    "rank_avg",
    "rating_avg",
    "rank_min",
    "rating_min",
    "matches_total",
    "matches_home",
    "matches_away",
    "matches_neutral",
    "wins",
    "losses",
    "draws",
    "goals_for",
    "goals_against",
    "confederation",
    "is_host",
]

OUTLIER_COLUMNS = [
    "win_rate",
    "goals_for_per_match",
    "goals_against_per_match",
    "goal_diff_per_match",
]


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    local_path = Path(path) if path is not None else None
    kaggle_error: Exception | None = None
    force_local = os.getenv("ELO_FORCE_LOCAL", "").strip().lower() in {"1", "true", "yes"}

    if not force_local and kagglehub is not None and KaggleDatasetAdapter is not None:
        try:
            data = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                KAGGLE_DATASET,
                KAGGLE_FILE,
            )
            source = "KaggleHub"
        except Exception as error:
            kaggle_error = error
            data = None
    else:
        data = None

    if data is None:
        if local_path is None or not local_path.exists():
            detail = f" Detalhe do KaggleHub: {kaggle_error}" if kaggle_error else ""
            raise FileNotFoundError(
                "Não foi possível carregar o dataset pelo KaggleHub e a cópia local não foi encontrada."
                + detail
            )
        data = pd.read_csv(local_path)
        source = "CSV local de contingência"

    missing_columns = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    data = data[RAW_COLUMNS].copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data.attrs["source"] = source
    return data


def validate_consistency(data: pd.DataFrame) -> dict[str, int]:
    return {
        "datas_invalidas": int(data["snapshot_date"].isna().sum()),
        "inconsistencias_mando_partidas": int(((data["matches_home"] + data["matches_away"] + data["matches_neutral"]) != data["matches_total"]).sum()),
        "inconsistencias_resultados_partidas": int(((data["wins"] + data["losses"] + data["draws"]) != data["matches_total"]).sum()),
        "partidas_nao_positivas": int((data["matches_total"] <= 0).sum()),
        "gols_negativos": int(((data["goals_for"] < 0) | (data["goals_against"] < 0)).sum()),
        "ranking_nao_positivo": int((data["rank"] <= 0).sum()),
    }


def remove_semantic_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    subset = [column for column in data.columns if column != "snapshot_date"]
    ordered = data.sort_values("snapshot_date", ascending=True).copy()
    duplicate_mask = ordered.duplicated(subset=subset, keep="first")
    return ordered.loc[~duplicate_mask].copy(), int(duplicate_mask.sum())


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["win_rate"] = result["wins"] / result["matches_total"]
    result["draw_rate"] = result["draws"] / result["matches_total"]
    result["loss_rate"] = result["losses"] / result["matches_total"]
    result["goals_for_per_match"] = result["goals_for"] / result["matches_total"]
    result["goals_against_per_match"] = result["goals_against"] / result["matches_total"]
    result["goal_diff_per_match"] = (result["goals_for"] - result["goals_against"]) / result["matches_total"]
    result["home_share"] = result["matches_home"] / result["matches_total"]
    result["away_share"] = result["matches_away"] / result["matches_total"]
    result["neutral_share"] = result["matches_neutral"] / result["matches_total"]
    result[TARGET_COLUMN] = (result["rank"] <= 10).astype(int)
    return result


def add_outlier_flags(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = data.copy()
    counts: dict[str, int] = {}
    flags: list[str] = []
    for column in OUTLIER_COLUMNS:
        q1 = float(result[column].quantile(0.25))
        q3 = float(result[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag = f"outlier_{column}"
        result[flag] = ~result[column].between(lower, upper, inclusive="both")
        counts[column] = int(result[flag].sum())
        flags.append(flag)
    result["outlier_any"] = result[flags].any(axis=1)
    counts["qualquer_variavel"] = int(result["outlier_any"].sum())
    return result, counts


def clean_data(raw_data: pd.DataFrame, minimum_matches: int = MINIMUM_MATCHES) -> tuple[pd.DataFrame, dict[str, Any]]:
    consistency = validate_consistency(raw_data)
    if any(consistency.values()):
        raise ValueError(f"Foram encontradas inconsistências estruturais: {consistency}")
    without_duplicates, semantic_removed = remove_semantic_duplicates(raw_data)
    low_experience_mask = without_duplicates["matches_total"] < minimum_matches
    filtered = without_duplicates.loc[~low_experience_mask].copy()
    engineered = add_engineered_features(filtered)
    cleaned, outlier_counts = add_outlier_flags(engineered)
    cleaned = cleaned.sort_values(["year", "rank", "country"]).reset_index(drop=True)
    summary: dict[str, Any] = {
        "origem_dados": str(raw_data.attrs.get("source", "não informada")),
        "linhas_iniciais": int(len(raw_data)),
        "colunas_originais": int(raw_data.shape[1]),
        "duplicatas_exatas_detectadas": int(raw_data.duplicated().sum()),
        "duplicatas_semanticas_removidas": semantic_removed,
        "criterio_minimo_partidas": int(minimum_matches),
        "registros_com_menos_de_30_partidas_removidos": int(low_experience_mask.sum()),
        "linhas_finais": int(len(cleaned)),
        "selecoes_distintas": int(cleaned["country"].nunique()),
        "ano_inicial": int(cleaned["year"].min()),
        "ano_final": int(cleaned["year"].max()),
        "ausentes_antes_do_tratamento": {column: int(value) for column, value in raw_data.isna().sum().items()},
        "ausentes_depois_do_tratamento": {column: int(value) for column, value in cleaned.isna().sum().items()},
        "verificacoes_de_consistencia": consistency,
        "outliers_sinalizados_e_mantidos": outlier_counts,
        "classe_elite_top10": {str(int(key)): int(value) for key, value in cleaned[TARGET_COLUMN].value_counts().sort_index().items()},
    }
    return cleaned, summary


def save_clean_data(data: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
