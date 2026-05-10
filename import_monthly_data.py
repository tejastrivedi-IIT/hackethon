from __future__ import annotations

import argparse
from pathlib import Path

from dt_analytics.config import DB_PATH
from dt_analytics.database import init_db, replace_period_data
from dt_analytics.processing import load_excel_data, normalize_monthly_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import month-year DT health reports into SQLite. Re-uploading the same period overwrites it."
    )
    parser.add_argument("files", nargs="+", help="Excel file paths downloaded from the MDM portal.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path.")
    parser.add_argument("--year", type=int, help="Override the year instead of reading it from the filename.")
    parser.add_argument("--month", type=int, choices=range(1, 13), help="Override the month number.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    init_db(args.db)
    total_rows = 0

    for file_name in args.files:
        file_path = Path(file_name).expanduser().resolve()
        raw_df = load_excel_data(file_path)
        normalized_df = normalize_monthly_dataframe(
            raw_df,
            file_path,
            year=args.year,
            month=args.month,
        )
        inserted_rows = replace_period_data(normalized_df, args.db)
        total_rows += inserted_rows
        print(
            f"Imported {inserted_rows} rows for {normalized_df['month_year_label'].iat[0]} "
            f"into {Path(args.db).resolve()}"
        )

    print(f"Finished. Total rows imported: {total_rows}")


if __name__ == "__main__":
    main()
