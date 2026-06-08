from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import clean_data, load_raw_data, save_clean_data

RAW_PATH = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "elo_ratings_wc2026_clean.csv"
SUMMARY_PATH = ROOT / "artifacts" / "resumo_limpeza.json"


def main() -> None:
    raw = load_raw_data(RAW_PATH)
    cleaned, summary = clean_data(raw)
    save_clean_data(cleaned, CLEAN_PATH)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Limpeza concluída.")
    print(f"Linhas originais: {len(raw)}")
    print(f"Linhas tratadas: {len(cleaned)}")
    print(f"Arquivo gerado: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
