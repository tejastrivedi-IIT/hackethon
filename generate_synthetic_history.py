from __future__ import annotations

from dt_analytics.synthetic import generate_and_append_synthetic_history


def main() -> None:
    result = generate_and_append_synthetic_history()
    print("Synthetic periods generated and appended:")
    for period in result.generated_periods:
        print(f"  {period}: {result.inserted_rows_by_period[period]} rows")
    print(f"Total synthetic rows appended: {result.total_rows}")


if __name__ == "__main__":
    main()
