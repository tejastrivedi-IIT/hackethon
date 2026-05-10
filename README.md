# DT Health Dashboard

This project turns month-year wise MDM DT health reports into:

- an offline Python import utility that loads each monthly Excel file into SQLite,
- overwrite-safe storage keyed by `period + dt_code`,
- a Streamlit dashboard with month/circle/division/subdivision filters,
- overload, unbalance, and off-hours analytics,
- next-year same-month DT failure risk prediction with leakage-aware training rules.

## Business rules used

The dashboard derives the two required fields instead of trusting the uploaded Excel values:

- `DT Loading = (Maximum KVA / KVA Rating) * 100`
- `DT Loading Status`
  - `low` if `< 20`
  - `normal` if `20 to 79.9`
  - `medium` if `80 to 100`
  - `high` if `> 100`
- `DT Unbalance Status`
  - compute `max(Unbalance(RY), Unbalance(YB), Unbalance(BR))`
  - `normal` if `<= 0.10`
  - `low` if `0.101 to 0.20`
  - `medium` if `0.201 to 0.30`
  - `high` if `> 0.30`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Import monthly files

```powershell
python import_monthly_data.py "DT Health Report May-2025.xlsx" "DT Health Report June-2025.xlsx"
```

If the same month-year file is imported again, that period is deleted and reloaded in SQLite, so the latest upload overwrites the old one.

If a filename does not contain the month and year, you can override it:

```powershell
python import_monthly_data.py "some_file.xlsx" --month 5 --year 2025
```

## Run the dashboard

```powershell
streamlit run app.py
```

## Generate synthetic backfill

This project also includes a reproducible synthetic history generator that uses the real May/June 2025 SQLite data as the anchor population and creates aligned records for:

- January 2025 to April 2025
- July 2025 to December 2025
- January 2026 to April 2026

```powershell
python generate_synthetic_history.py
```

The script keeps the existing real May/June 2025 data untouched and appends or replaces only the synthetic periods listed above.

## Leakage-safe prediction design

There is no actual DT failure label in the uploaded MDM files, so the app predicts a **failure proxy** for the same DT in the same month next year.

- Target creation uses only future records from `current month + 12 months`.
- Features use only the current record and lagged history from earlier months.
- Validation uses later target periods as a time-separated holdout.
- If the SQLite database does not yet contain enough next-year labels, the app automatically falls back to a transparent heuristic stress score instead of training a leakage-prone model.

This makes the demo honest for the hackathon now, while still allowing a stronger supervised model later if more monthly history is imported.
