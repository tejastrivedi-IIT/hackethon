from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DB_PATH


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | Path = DB_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as connection:
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(dt_monthly_health)").fetchall()
        }
        if existing_columns and "source_row_number" not in existing_columns:
            connection.execute("DROP TABLE IF EXISTS dt_monthly_health")
            connection.execute("DROP TABLE IF EXISTS import_history")

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS dt_monthly_health (
                period TEXT NOT NULL,
                month_year_label TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                source_row_number INTEGER NOT NULL,
                zone TEXT,
                circle TEXT,
                division TEXT,
                subdivision TEXT,
                substation TEXT,
                feeder_name TEXT,
                dt_code TEXT NOT NULL,
                dt_meter_number TEXT,
                dt_name TEXT,
                hes TEXT,
                mf REAL,
                kva_rating REAL,
                kwh REAL,
                kvah REAL,
                power_factor REAL,
                avg_kva REAL,
                maximum_kva REAL,
                maximum_kva_date TEXT,
                r_phase_max_current REAL,
                y_phase_max_current REAL,
                b_phase_max_current REAL,
                unbalance_ry REAL,
                unbalance_yb REAL,
                unbalance_br REAL,
                max_unbalance REAL,
                dt_loading_source REAL,
                dt_loading REAL,
                load_factor REAL,
                utilization_factor REAL,
                dt_loading_status_source TEXT,
                dt_loading_status TEXT,
                dt_unbalance_status_source TEXT,
                dt_unbalance_status TEXT,
                loading_status_score INTEGER,
                unbalance_status_score INTEGER,
                total_hours REAL,
                power_off_hours REAL,
                power_on_hours REAL,
                power_off_ratio REAL,
                criticality_score REAL,
                import_file TEXT,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (period, source_row_number)
            );

            CREATE TABLE IF NOT EXISTS import_history (
                period TEXT NOT NULL PRIMARY KEY,
                month_year_label TEXT NOT NULL,
                source_file TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_dt_health_period ON dt_monthly_health (period);
            CREATE INDEX IF NOT EXISTS idx_dt_health_month ON dt_monthly_health (month);
            CREATE INDEX IF NOT EXISTS idx_dt_health_circle ON dt_monthly_health (circle);
            CREATE INDEX IF NOT EXISTS idx_dt_health_division ON dt_monthly_health (division);
            CREATE INDEX IF NOT EXISTS idx_dt_health_subdivision ON dt_monthly_health (subdivision);
            CREATE INDEX IF NOT EXISTS idx_dt_health_dt_code ON dt_monthly_health (dt_code);
            """
        )


def replace_period_data(df: pd.DataFrame, db_path: str | Path = DB_PATH) -> int:
    if df.empty:
        return 0

    init_db(db_path)
    period = df["period"].iat[0]
    label = df["month_year_label"].iat[0]
    source_file = df["import_file"].iat[0]

    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM dt_monthly_health WHERE period = ?", (period,))
        df.to_sql("dt_monthly_health", connection, if_exists="append", index=False)
        connection.execute(
            """
            INSERT INTO import_history (period, month_year_label, source_file, row_count, imported_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(period) DO UPDATE SET
                month_year_label = excluded.month_year_label,
                source_file = excluded.source_file,
                row_count = excluded.row_count,
                imported_at = excluded.imported_at
            """,
            (period, label, source_file, int(len(df))),
        )
        connection.commit()

    return int(len(df))


def fetch_all_data(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM dt_monthly_health
            ORDER BY year DESC, month DESC, circle, division, subdivision, dt_code
            """,
            connection,
            parse_dates=["maximum_kva_date", "imported_at"],
        )


def fetch_period_data(period: str, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM dt_monthly_health
            WHERE period = ?
            ORDER BY circle, division, subdivision, dt_code, source_row_number
            """,
            connection,
            params=(period,),
            parse_dates=["maximum_kva_date", "imported_at"],
        )


def fetch_month_history_data(month: int, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM dt_monthly_health
            WHERE month = ?
            ORDER BY year DESC, circle, division, subdivision, dt_code, source_row_number
            """,
            connection,
            params=(int(month),),
            parse_dates=["maximum_kva_date", "imported_at"],
        )


def fetch_month_model_data(month: int, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                period,
                year,
                month,
                source_row_number,
                circle,
                division,
                subdivision,
                dt_code,
                dt_meter_number,
                dt_name,
                kva_rating,
                kwh,
                kvah,
                power_factor,
                avg_kva,
                maximum_kva,
                r_phase_max_current,
                y_phase_max_current,
                b_phase_max_current,
                unbalance_ry,
                unbalance_yb,
                unbalance_br,
                max_unbalance,
                dt_loading,
                load_factor,
                utilization_factor,
                loading_status_score,
                unbalance_status_score,
                total_hours,
                power_off_hours,
                power_on_hours,
                power_off_ratio,
                criticality_score
            FROM dt_monthly_health
            WHERE month = ?
            ORDER BY year DESC, circle, division, subdivision, dt_code, source_row_number
            """,
            connection,
            params=(int(month),),
        )


def fetch_import_history(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT period, month_year_label, source_file, row_count, imported_at
            FROM import_history
            ORDER BY period DESC
            """,
            connection,
            parse_dates=["imported_at"],
        )
