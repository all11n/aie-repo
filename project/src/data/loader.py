# src/data/loader.py
"""Модуль для загрузки данных."""
import pandas as pd
from pathlib import Path
from src.data.preprocessing import convert_to_multilabel, DISEASE_LABELS

def load_cardio_data(filepath: str = "data/cardio.csv"):
    candidates = [
        Path(filepath),
        Path("..") / filepath,
        Path("../..") / filepath,
        Path.home() / "doc" / "aie-repo" / "project" / "data" / "cardio.csv",
    ]

    full_path = None
    for cand in candidates:
        if cand.exists():
            full_path = cand
            break

    if full_path is None:
        raise FileNotFoundError(
            f"Файл '{filepath}' не найден ни в одном из мест:\n" +
            "\n".join(f"  • {p}" for p in candidates)
        )

    df = pd.read_csv(full_path, delimiter=';')
    print(f"Загружено {len(df)} записей из {full_path}")
    return df

def load_data(filepath: str = "data/cardio.csv"):
    df_original = load_cardio_data(filepath)
    df_multilabel = convert_to_multilabel(df_original)
    print(f"Преобразовано в мультилейбл: {len(df_multilabel)} записей")
    return df_multilabel