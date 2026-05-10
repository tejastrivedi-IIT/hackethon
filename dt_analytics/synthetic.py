from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH
from .database import fetch_all_data, replace_period_data
from .processing import derive_loading_status, derive_unbalance_status

DESCRIPTOR_COLUMNS = [
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
]

MONTHS_TO_GENERATE = [
    (2025, 1),
    (2025, 2),
    (2025, 3),
    (2025, 4),
    (2025, 7),
    (2025, 8),
    (2025, 9),
    (2025, 10),
    (2025, 11),
    (2025, 12),
    (2026, 1),
    (2026, 2),
    (2026, 3),
    (2026, 4),
]

LOAD_MULTIPLIER = {
    1: 0.90,
    2: 0.92,
    3: 0.99,
    4: 1.06,
    5: 1.10,
    6: 1.07,
    7: 0.98,
    8: 0.96,
    9: 0.95,
    10: 0.97,
    11: 0.94,
    12: 0.92,
}

UNBALANCE_MULTIPLIER = {
    1: 0.94,
    2: 0.95,
    3: 0.98,
    4: 1.02,
    5: 1.00,
    6: 1.01,
    7: 1.03,
    8: 1.03,
    9: 1.01,
    10: 0.99,
    11: 0.97,
    12: 0.95,
}

OFF_HOURS_MULTIPLIER = {
    1: 0.55,
    2: 0.58,
    3: 0.70,
    4: 0.84,
    5: 1.00,
    6: 1.14,
    7: 1.18,
    8: 1.16,
    9: 1.07,
    10: 0.88,
    11: 0.76,
    12: 0.66,
}

ENERGY_MULTIPLIER = {
    1: 0.89,
    2: 0.91,
    3: 0.98,
    4: 1.05,
    5: 1.09,
    6: 1.06,
    7: 0.99,
    8: 0.97,
    9: 0.96,
    10: 0.98,
    11: 0.94,
    12: 0.91,
}


@dataclass
class SyntheticGenerationResult:
    generated_periods: list[str]
    inserted_rows_by_period: dict[str, int]
    total_rows: int


def _month_label(year: int, month: int) -> str:
    return f"{calendar.month_name[month]} {year}"


def _period_string(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _stable_period_rng(period: str) -> np.random.Generator:
    digest = hashlib.sha256(period.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _prepare_real_panel(real_df: pd.DataFrame) -> pd.DataFrame:
    panel = real_df.copy()
    panel = panel.sort_values(["period"] + DESCRIPTOR_COLUMNS + ["source_row_number"]).reset_index(drop=True)
    panel["dup_rank"] = panel.groupby(["period"] + DESCRIPTOR_COLUMNS, dropna=False).cumcount() + 1
    panel["entity_key"] = (
        panel[DESCRIPTOR_COLUMNS]
        .astype("string")
        .fillna("")
        .agg("|".join, axis=1)
        + "|"
        + panel["dup_rank"].astype(str)
    )
    return panel


def _build_baseline_entity_frame(real_df: pd.DataFrame) -> pd.DataFrame:
    panel = _prepare_real_panel(real_df)

    clipped = panel.copy()
    clipped["dt_loading_capped"] = clipped["dt_loading"].clip(lower=0, upper=350)
    clipped["max_unbalance_capped"] = clipped["max_unbalance"].clip(lower=0, upper=1)
    clipped["power_off_hours_capped"] = clipped["power_off_hours"].clip(lower=0).where(
        clipped["total_hours"].notna(),
        clipped["power_off_hours"].clip(lower=0),
    )
    clipped["power_off_hours_capped"] = clipped[["power_off_hours_capped", "total_hours"]].min(axis=1)
    clipped["power_factor_capped"] = clipped["power_factor"].clip(lower=0.72, upper=0.999)

    for column in ["kwh", "kvah", "avg_kva", "maximum_kva"]:
        upper = float(clipped[column].quantile(0.995))
        clipped[f"{column}_capped"] = clipped[column].clip(lower=0, upper=upper)

    clipped["avg_to_max_ratio"] = (
        clipped["avg_kva_capped"] / clipped["maximum_kva_capped"].replace(0, np.nan)
    ).clip(lower=0.10, upper=0.95)
    clipped["current_per_kva"] = (
        clipped[["r_phase_max_current", "y_phase_max_current", "b_phase_max_current"]]
        .mean(axis=1)
        / clipped["maximum_kva_capped"].replace(0, np.nan)
    ).clip(lower=0.20, upper=8.0)

    grouped = clipped.groupby("entity_key", dropna=False)
    baseline = grouped[DESCRIPTOR_COLUMNS].last()

    baseline["baseline_dt_loading"] = grouped.apply(
        lambda g: (g["dt_loading_capped"] / g["month"].map(LOAD_MULTIPLIER)).median()
    )
    baseline["baseline_unbalance"] = grouped.apply(
        lambda g: (g["max_unbalance_capped"] / g["month"].map(UNBALANCE_MULTIPLIER)).median()
    )
    baseline["baseline_off_hours"] = grouped.apply(
        lambda g: (g["power_off_hours_capped"] / g["month"].map(OFF_HOURS_MULTIPLIER)).median()
    )
    baseline["baseline_kwh"] = grouped.apply(
        lambda g: (g["kwh_capped"] / g["month"].map(ENERGY_MULTIPLIER)).median()
    )
    baseline["baseline_kvah"] = grouped.apply(
        lambda g: (g["kvah_capped"] / g["month"].map(ENERGY_MULTIPLIER)).median()
    )
    baseline["baseline_pf"] = grouped["power_factor_capped"].median()
    baseline["baseline_avg_to_max_ratio"] = grouped["avg_to_max_ratio"].median().fillna(0.42)
    baseline["baseline_current_per_kva"] = grouped["current_per_kva"].median().fillna(1.25)
    baseline["baseline_mf"] = grouped["mf"].median().fillna(1.0)

    ratios = grouped[["unbalance_ry", "unbalance_yb", "unbalance_br"]].median().div(
        grouped["max_unbalance_capped"].median().replace(0, np.nan),
        axis=0,
    )
    ratios = ratios.clip(lower=0.10, upper=1.0).fillna(0.72)
    baseline["ratio_ry"] = ratios["unbalance_ry"]
    baseline["ratio_yb"] = ratios["unbalance_yb"]
    baseline["ratio_br"] = ratios["unbalance_br"]

    circle_loading_baseline = baseline.groupby("circle")["baseline_dt_loading"].transform("median")
    circle_unbalance_baseline = baseline.groupby("circle")["baseline_unbalance"].transform("median")
    circle_off_baseline = baseline.groupby("circle")["baseline_off_hours"].transform("median")
    circle_energy_baseline = baseline.groupby("circle")["baseline_kwh"].transform("median")

    overall_loading = float(baseline["baseline_dt_loading"].median())
    overall_unbalance = float(baseline["baseline_unbalance"].median())
    overall_off = float(baseline["baseline_off_hours"].median())
    overall_energy = float(baseline["baseline_kwh"].median())

    baseline["circle_load_bias"] = (circle_loading_baseline / overall_loading).fillna(1.0).clip(0.80, 1.25)
    baseline["circle_unbalance_bias"] = (
        circle_unbalance_baseline / overall_unbalance
    ).fillna(1.0).clip(0.85, 1.20)
    baseline["circle_off_bias"] = (circle_off_baseline / overall_off).fillna(1.0).clip(0.55, 1.60)
    baseline["circle_energy_bias"] = (circle_energy_baseline / overall_energy).fillna(1.0).clip(0.80, 1.25)

    baseline = baseline.reset_index()
    baseline["baseline_dt_loading"] = baseline["baseline_dt_loading"].fillna(overall_loading).clip(0, 350)
    baseline["baseline_unbalance"] = baseline["baseline_unbalance"].fillna(overall_unbalance).clip(0, 1)
    baseline["baseline_off_hours"] = baseline["baseline_off_hours"].fillna(overall_off).clip(0, 350)
    baseline["baseline_kwh"] = baseline["baseline_kwh"].fillna(overall_energy).clip(0)
    baseline["baseline_kvah"] = baseline["baseline_kvah"].fillna(baseline["baseline_kwh"] * 1.03).clip(0)
    baseline["baseline_pf"] = baseline["baseline_pf"].fillna(0.94).clip(0.72, 0.999)
    baseline["baseline_avg_to_max_ratio"] = baseline["baseline_avg_to_max_ratio"].fillna(0.42).clip(0.10, 0.95)
    baseline["baseline_current_per_kva"] = baseline["baseline_current_per_kva"].fillna(1.25).clip(0.20, 8.0)
    baseline["mf"] = baseline["mf"].fillna(baseline["baseline_mf"]).fillna(1.0)
    return baseline


def _make_month_dataframe(baseline: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    period = _period_string(year, month)
    month_label = _month_label(year, month)
    total_hours = calendar.monthrange(year, month)[1] * 24
    rng = _stable_period_rng(period)
    generated = baseline.copy()

    months_from_may_2025 = (year - 2025) * 12 + (month - 5)
    aging_factor = 1 + (months_from_may_2025 * 0.0025)
    outage_factor = 1 + max(months_from_may_2025, 0) * 0.0015

    load_noise = rng.normal(0, 0.055, len(generated))
    unbalance_noise = rng.normal(0, 0.040, len(generated))
    off_noise = rng.normal(0, 0.22, len(generated))
    pf_noise = rng.normal(0, 0.010, len(generated))
    ratio_noise = rng.normal(0, 0.035, len(generated))
    current_noise = rng.normal(0, 0.05, (len(generated), 3))
    date_days = rng.integers(1, calendar.monthrange(year, month)[1] + 1, len(generated))

    generated["period"] = period
    generated["month_year_label"] = month_label
    generated["year"] = year
    generated["month"] = month
    generated["source_row_number"] = np.arange(2, len(generated) + 2)
    generated["maximum_kva_date"] = pd.to_datetime(
        {
            "year": np.full(len(generated), year),
            "month": np.full(len(generated), month),
            "day": date_days,
        }
    )

    load_multiplier = LOAD_MULTIPLIER[month] * max(aging_factor, 0.92)
    unbalance_multiplier = UNBALANCE_MULTIPLIER[month] * max(1 + (months_from_may_2025 * 0.001), 0.95)
    off_multiplier = OFF_HOURS_MULTIPLIER[month] * max(outage_factor, 0.90)
    energy_multiplier = ENERGY_MULTIPLIER[month] * max(aging_factor, 0.93)

    generated["dt_loading"] = (
        generated["baseline_dt_loading"]
        * generated["circle_load_bias"]
        * load_multiplier
        * (1 + load_noise)
    ).clip(lower=0, upper=350).round(1)
    generated["maximum_kva"] = ((generated["dt_loading"] / 100.0) * generated["kva_rating"]).round(3)

    generated["max_unbalance"] = (
        generated["baseline_unbalance"]
        * generated["circle_unbalance_bias"]
        * unbalance_multiplier
        * (1 + unbalance_noise)
    ).clip(lower=0, upper=1.0).round(3)

    base_phase = generated["max_unbalance"].replace(0, 0.001)
    generated["unbalance_ry"] = (base_phase * (generated["ratio_ry"] * (1 + rng.normal(0, 0.06, len(generated))))).clip(
        lower=0, upper=1.0
    )
    generated["unbalance_yb"] = (base_phase * (generated["ratio_yb"] * (1 + rng.normal(0, 0.06, len(generated))))).clip(
        lower=0, upper=1.0
    )
    generated["unbalance_br"] = (base_phase * (generated["ratio_br"] * (1 + rng.normal(0, 0.06, len(generated))))).clip(
        lower=0, upper=1.0
    )
    generated["unbalance_ry"] = generated[["unbalance_ry", "max_unbalance"]].max(axis=1).round(3)
    generated["unbalance_yb"] = generated["unbalance_yb"].round(3)
    generated["unbalance_br"] = generated["unbalance_br"].round(3)
    generated["max_unbalance"] = generated[["unbalance_ry", "unbalance_yb", "unbalance_br"]].max(axis=1).round(3)

    generated["power_off_hours"] = (
        generated["baseline_off_hours"]
        * generated["circle_off_bias"]
        * off_multiplier
        * (1 + off_noise)
    ).clip(lower=0, upper=total_hours * 0.85).round(1)
    generated["total_hours"] = float(total_hours)
    generated["power_on_hours"] = (generated["total_hours"] - generated["power_off_hours"]).clip(lower=0).round(1)
    generated["power_off_ratio"] = (generated["power_off_hours"] / generated["total_hours"]).round(4)

    generated["power_factor"] = (
        generated["baseline_pf"]
        - (generated["max_unbalance"] - generated["baseline_unbalance"]) * 0.03
        + pf_noise
    ).clip(lower=0.72, upper=0.999).round(3)

    generated["kwh"] = (
        generated["baseline_kwh"]
        * generated["circle_energy_bias"]
        * energy_multiplier
        * (1 + rng.normal(0, 0.07, len(generated)))
    ).clip(lower=0).round(3)
    generated["kvah"] = (generated["kwh"] / generated["power_factor"].replace(0, 0.9)).clip(
        lower=generated["kwh"], upper=generated["kwh"] * 1.35
    ).round(3)

    avg_ratio = (generated["baseline_avg_to_max_ratio"] * (1 + ratio_noise)).clip(lower=0.10, upper=0.95)
    generated["avg_kva"] = (generated["maximum_kva"] * avg_ratio).round(3)
    generated["load_factor"] = ((generated["avg_kva"] / generated["maximum_kva"].replace(0, np.nan)) * 100).fillna(0).clip(
        lower=0, upper=100
    ).round(1)
    generated["utilization_factor"] = generated["dt_loading"].clip(lower=0).round(1)

    base_current = (generated["maximum_kva"] * generated["baseline_current_per_kva"]).clip(lower=0)
    generated["r_phase_max_current"] = (base_current * (1 + current_noise[:, 0] + (generated["max_unbalance"] * 0.10))).clip(
        lower=0
    ).round(2)
    generated["y_phase_max_current"] = (base_current * (1 + current_noise[:, 1] - (generated["max_unbalance"] * 0.05))).clip(
        lower=0
    ).round(2)
    generated["b_phase_max_current"] = (base_current * (1 + current_noise[:, 2] - (generated["max_unbalance"] * 0.05))).clip(
        lower=0
    ).round(2)

    generated["dt_loading_source"] = generated["dt_loading"]
    generated["dt_loading_status"] = derive_loading_status(generated["dt_loading"]).fillna("normal")
    generated["dt_loading_status_source"] = generated["dt_loading_status"]
    generated["dt_unbalance_status"] = derive_unbalance_status(generated["max_unbalance"]).fillna("normal")
    generated["dt_unbalance_status_source"] = generated["dt_unbalance_status"]
    generated["loading_status_score"] = generated["dt_loading_status"].map(
        {"normal": 0, "low": 1, "medium": 2, "high": 3}
    ).fillna(0).astype(int)
    generated["unbalance_status_score"] = generated["dt_unbalance_status"].map(
        {"normal": 0, "low": 1, "medium": 2, "high": 3}
    ).fillna(0).astype(int)
    generated["criticality_score"] = (
        (generated["loading_status_score"] * 0.45)
        + (generated["unbalance_status_score"] * 0.40)
        + (generated["power_off_ratio"].clip(0, 1) * 0.15 * 3)
    ).round(3)
    generated["import_file"] = "synthetic_backfill_from_may_june_2025"

    output_columns = [
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
    return generated[output_columns]


def generate_and_append_synthetic_history(
    db_path: str | Path = DB_PATH,
    months_to_generate: list[tuple[int, int]] | None = None,
) -> SyntheticGenerationResult:
    all_df = fetch_all_data(db_path)
    real_df = all_df[all_df["import_file"] != "synthetic_backfill_from_may_june_2025"].copy()
    if real_df.empty:
        raise ValueError("SQLite database is empty. Import at least one real month before generating synthetic history.")

    baseline = _build_baseline_entity_frame(real_df)
    periods = months_to_generate or MONTHS_TO_GENERATE

    inserted_rows_by_period: dict[str, int] = {}
    generated_periods: list[str] = []

    for year, month in periods:
        month_df = _make_month_dataframe(baseline, year, month)
        row_count = replace_period_data(month_df, db_path)
        period = _period_string(year, month)
        inserted_rows_by_period[period] = row_count
        generated_periods.append(period)

    return SyntheticGenerationResult(
        generated_periods=generated_periods,
        inserted_rows_by_period=inserted_rows_by_period,
        total_rows=sum(inserted_rows_by_period.values()),
    )
