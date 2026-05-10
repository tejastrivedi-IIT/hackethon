from __future__ import annotations

import calendar
import re
from pathlib import Path

import pandas as pd

from .config import EXPECTED_COLUMNS, NUMERIC_COLUMNS, RENAME_MAP, STATUS_SCORES, TEXT_COLUMNS

MONTH_PATTERN = re.compile(
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[\s_-]*(?P<year>20\d{2})",
    re.IGNORECASE,
)


def parse_period_from_filename(file_path: str | Path) -> tuple[int, int]:
    match = MONTH_PATTERN.search(Path(file_path).stem)
    if not match:
        raise ValueError(
            "Could not detect month-year from filename. Use a name like 'DT Health Report May-2025.xlsx'."
        )

    month_text = match.group("month").strip().lower()
    month_lookup = {name.lower(): idx for idx, name in enumerate(calendar.month_name) if idx}
    month_lookup.update({name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if idx})
    month = month_lookup[month_text[:3].title().lower()] if month_text not in month_lookup else month_lookup[month_text]
    year = int(match.group("year"))
    return year, month


def load_excel_data(file_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(file_path, engine="openpyxl")


def derive_loading_status(series: pd.Series) -> pd.Series:
    bins = [-float("inf"), 19.9, 79.9, 100.0, float("inf")]
    labels = ["low", "normal", "medium", "high"]
    return pd.cut(series.fillna(-1), bins=bins, labels=labels, include_lowest=True).astype("string")


def derive_unbalance_status(series: pd.Series) -> pd.Series:
    bins = [-float("inf"), 0.1, 0.2, 0.3, float("inf")]
    labels = ["normal", "low", "medium", "high"]
    return pd.cut(series.fillna(-1), bins=bins, labels=labels, include_lowest=True).astype("string")


def _month_name(month: int) -> str:
    return calendar.month_name[month]


def normalize_monthly_dataframe(
    raw_df: pd.DataFrame,
    file_path: str | Path,
    *,
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    missing = [column for column in EXPECTED_COLUMNS if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if year is None or month is None:
        parsed_year, parsed_month = parse_period_from_filename(file_path)
        year = year or parsed_year
        month = month or parsed_month

    df = raw_df[EXPECTED_COLUMNS].rename(columns=RENAME_MAP).copy()

    for column in TEXT_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["maximum_kva_date"] = pd.to_datetime(df["maximum_kva_date"], errors="coerce")
    df["year"] = int(year)
    df["month"] = int(month)
    df["period"] = f"{year:04d}-{month:02d}"
    df["month_year_label"] = f"{_month_name(month)} {year}"
    df["import_file"] = Path(file_path).name
    df["source_row_number"] = range(2, len(df) + 2)

    df["dt_loading"] = ((df["maximum_kva"] / df["kva_rating"]) * 100).round(1)
    df["max_unbalance"] = df[["unbalance_ry", "unbalance_yb", "unbalance_br"]].max(axis=1)
    df["dt_loading_status"] = derive_loading_status(df["dt_loading"]).fillna("normal")
    df["dt_unbalance_status"] = derive_unbalance_status(df["max_unbalance"]).fillna("normal")

    df["loading_status_score"] = df["dt_loading_status"].map(STATUS_SCORES).fillna(0).astype(int)
    df["unbalance_status_score"] = df["dt_unbalance_status"].map(STATUS_SCORES).fillna(0).astype(int)
    df["power_off_ratio"] = (df["power_off_hours"] / df["total_hours"]).fillna(0)
    df["criticality_score"] = (
        (df["loading_status_score"] * 0.45)
        + (df["unbalance_status_score"] * 0.40)
        + (df["power_off_ratio"].clip(0, 1) * 0.15 * 3)
    ).round(3)

    ordered_columns = [
        "period",
        "month_year_label",
        "year",
        "month",
        "source_row_number",
        "zone",
        "circle",
        "division",
        "subdivision",
        "substation",
        "feeder_name",
        "dt_code",
        "dt_meter_number",
        "dt_name",
        "hes",
        "mf",
        "kva_rating",
        "kwh",
        "kvah",
        "power_factor",
        "avg_kva",
        "maximum_kva",
        "maximum_kva_date",
        "r_phase_max_current",
        "y_phase_max_current",
        "b_phase_max_current",
        "unbalance_ry",
        "unbalance_yb",
        "unbalance_br",
        "max_unbalance",
        "dt_loading_source",
        "dt_loading",
        "load_factor",
        "utilization_factor",
        "dt_loading_status_source",
        "dt_loading_status",
        "dt_unbalance_status_source",
        "dt_unbalance_status",
        "loading_status_score",
        "unbalance_status_score",
        "total_hours",
        "power_off_hours",
        "power_on_hours",
        "power_off_ratio",
        "criticality_score",
        "import_file",
    ]

    return df[ordered_columns]
