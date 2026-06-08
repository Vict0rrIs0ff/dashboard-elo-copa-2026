from pathlib import Path

from src.data_processing import clean_data, load_raw_data

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"


def test_raw_dataset_has_expected_shape() -> None:
    raw = load_raw_data(RAW_PATH)
    assert raw.shape == (4683, 23)


def test_cleaning_removes_semantic_duplicates_and_short_histories() -> None:
    raw = load_raw_data(RAW_PATH)
    cleaned, summary = clean_data(raw)
    assert summary["duplicatas_semanticas_removidas"] == 48
    assert summary["registros_com_menos_de_30_partidas_removidos"] == 728
    assert len(cleaned) == 3907


def test_cleaned_dataset_has_no_missing_values() -> None:
    raw = load_raw_data(RAW_PATH)
    cleaned, _ = clean_data(raw)
    assert int(cleaned.isna().sum().sum()) == 0


def test_match_counts_are_consistent() -> None:
    raw = load_raw_data(RAW_PATH)
    cleaned, _ = clean_data(raw)
    assert int(((cleaned["wins"] + cleaned["losses"] + cleaned["draws"]) != cleaned["matches_total"]).sum()) == 0
    assert int(((cleaned["matches_home"] + cleaned["matches_away"] + cleaned["matches_neutral"]) != cleaned["matches_total"]).sum()) == 0


def test_target_is_binary() -> None:
    raw = load_raw_data(RAW_PATH)
    cleaned, _ = clean_data(raw)
    assert sorted(cleaned["elite_top10"].unique().tolist()) == [0, 1]
