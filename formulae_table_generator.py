from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# Columns of the formulae table CSV, in display/save order.
FORMULA_TABLE_COLUMNS = [
    "topic",
    "subtopic",
    "name",
    "equation_latex",
    "description_from_image",
    "formula_in_plain_english",
    "description_from_gemini",
    "sympy_formula",
]

# Old column names mapped onto the current schema when loading legacy CSVs.
LEGACY_COLUMN_ALIASES = {"description_text": "description_from_image"}

DEFAULT_TABLE_PATH = "csv_tables/formulae_table.csv"


def empty_row() -> dict[str, str]:
    return {column: "" for column in FORMULA_TABLE_COLUMNS}


def _coerce_record(row: Any) -> dict[str, str] | None:
    if not isinstance(row, dict):
        return None
    normalized_row = empty_row()
    for column in FORMULA_TABLE_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            for legacy, modern in LEGACY_COLUMN_ALIASES.items():
                if modern == column and row.get(legacy):
                    value = row.get(legacy)
                    break
        normalized_row[column] = str(value or "")
    return normalized_row


def normalize_table_rows(rows: Any) -> list[dict[str, str]]:
    if hasattr(rows, "to_dict"):
        records = rows.to_dict("records")
    else:
        records = list(rows or [])

    normalized: list[dict[str, str]] = []
    for row in records:
        coerced = _coerce_record(row)
        if coerced is None:
            continue
        if any(value.strip() for value in coerced.values()):
            normalized.append(coerced)
    return normalized


def _resolve_path(destination: str) -> Path:
    path = Path(destination).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_formula_table(path: str = DEFAULT_TABLE_PATH) -> list[dict[str, str]]:
    table_path = _resolve_path(path)
    if not table_path.exists():
        return []
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return normalize_table_rows(list(reader))
