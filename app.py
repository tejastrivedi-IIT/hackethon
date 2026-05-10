from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dt_analytics.config import BASE_DIR, DB_PATH, EXPORT_COLUMNS
from dt_analytics import database as db
from dt_analytics import modeling as mdl
from dt_analytics.processing import load_excel_data, normalize_monthly_dataframe
from dt_analytics.synthetic import generate_and_append_synthetic_history


st.set_page_config(page_title="DT Health Command Center", layout="wide")

BOOTSTRAP_REAL_FILES = [
    "DT Health Report May-2025.xlsx",
    "DT Health Report June-2025.xlsx",
]

def get_db_signature() -> tuple[str, int]:
    path = Path(DB_PATH)
    if not path.exists():
        return str(path.resolve()), 0
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns


def bootstrap_synthetic_months() -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in (2023, 2024):
        for month in range(1, 13):
            months.append((year, month))
    for month in [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]:
        months.append((2025, month))
    for month in range(1, 5):
        months.append((2026, month))
    return months


def ensure_database_ready() -> None:
    db.init_db(DB_PATH)
    if not db.fetch_import_history(DB_PATH).empty:
        return

    missing_files = [name for name in BOOTSTRAP_REAL_FILES if not (BASE_DIR / name).exists()]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Bundled bootstrap files are missing from the app repository: {missing_text}"
        )

    for file_name in BOOTSTRAP_REAL_FILES:
        file_path = BASE_DIR / file_name
        raw_df = load_excel_data(file_path)
        monthly_df = normalize_monthly_dataframe(raw_df, file_path)
        db.replace_period_data(monthly_df, DB_PATH)

    generate_and_append_synthetic_history(DB_PATH, months_to_generate=bootstrap_synthetic_months())


@st.cache_data(show_spinner=False)
def load_import_history_data(_db_signature: tuple[str, int]) -> pd.DataFrame:
    return db.fetch_import_history(DB_PATH)


@st.cache_data(show_spinner=False)
def load_period_dashboard_data(period: str, _db_signature: tuple[str, int]) -> pd.DataFrame:
    return db.fetch_period_data(period, DB_PATH)


@st.cache_data(show_spinner=False)
def load_period_predictions(period: str, _db_signature: tuple[str, int]) -> tuple[pd.DataFrame, dict]:
    month = int(str(period).split("-")[1])
    df = db.fetch_month_model_data(month, DB_PATH)
    result = mdl.attach_failure_predictions_for_period(df, period)
    prediction_df = result.scored_df[
        ["period", "source_row_number", "prediction_target_period", "predicted_failure_probability", "prediction_method"]
    ].copy()
    info = {
        "method": result.method,
        "model_name": "SGD Logistic Regression Classifier (same-calendar-month historical model)"
        if result.method == "model"
        else "Stress-score heuristic",
        "detail": result.detail,
        "training_rows": result.training_rows,
        "positive_rows": result.positive_rows,
        "validation_auc": result.validation_auc,
    }
    return prediction_df, info


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 214, 171, 0.38), transparent 28%),
                radial-gradient(circle at top right, rgba(164, 214, 196, 0.35), transparent 24%),
                linear-gradient(180deg, #fbf7ef 0%, #f5efe1 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .hero {
            padding: 1.45rem 1.65rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #17324d 0%, #305f72 55%, #c36d43 100%);
            color: #fffaf0;
            box-shadow: 0 18px 38px rgba(31, 51, 71, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            letter-spacing: 0.01em;
        }
        .hero p {
            margin: 0.15rem 0;
            font-size: 1rem;
            line-height: 1.5;
        }
        .badge-row {
            margin-top: 0.85rem;
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 250, 240, 0.18);
            border: 1px solid rgba(255, 250, 240, 0.2);
            font-size: 0.9rem;
        }
        .section-card {
            padding: 1rem 1.1rem 1.1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 252, 245, 0.92);
            border: 1px solid rgba(89, 111, 125, 0.12);
            box-shadow: 0 10px 22px rgba(80, 88, 94, 0.08);
            margin-bottom: 0.9rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #17324d;
            margin-bottom: 0.55rem;
        }
        .subtle {
            color: #5c6670;
            font-size: 0.92rem;
        }
        .metric-card {
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            color: white;
            box-shadow: 0 14px 28px rgba(53, 64, 78, 0.16);
            min-height: 175px;
            margin-bottom: 0.25rem;
        }
        .metric-card h3 {
            font-size: 0.95rem;
            margin: 0 0 0.45rem 0;
            font-weight: 600;
            opacity: 0.95;
        }
        .metric-number {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.45rem;
        }
        .metric-note {
            font-size: 0.88rem;
            line-height: 1.35;
            opacity: 0.92;
        }
        .metric-total { background: linear-gradient(135deg, #355c7d 0%, #17324d 100%); }
        .metric-overload { background: linear-gradient(135deg, #d65a31 0%, #7a2d16 100%); }
        .metric-unbalance { background: linear-gradient(135deg, #2f6f6d 0%, #163a39 100%); }
        .metric-offhours { background: linear-gradient(135deg, #8a5a44 0%, #4c2f24 100%); }
        .prediction-panel {
            border-radius: 22px;
            padding: 1rem 1.1rem;
            color: #17324d;
            background: linear-gradient(180deg, #fff7e9 0%, #fde9c7 100%);
            border: 1px solid rgba(195, 109, 67, 0.28);
            box-shadow: 0 12px 26px rgba(137, 98, 56, 0.10);
        }
        .prediction-chip {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            color: #fffdf8;
        }
        .chip-high { background: #b14522; }
        .chip-medium { background: #bf7c14; }
        .chip-low { background: #2e7d5a; }
        .summary-tip {
            padding: 0.8rem 0.95rem;
            border-left: 5px solid #c36d43;
            background: rgba(255, 249, 239, 0.95);
            border-radius: 12px;
            color: #5f4d42;
            margin-bottom: 0.7rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: #17324d;
            color: #fffaf2;
            font-weight: 600;
            padding: 0.55rem 0.8rem;
        }
        .stButton>button:hover {
            background: #214869;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_filters(period_df: pd.DataFrame, history: pd.DataFrame, selected_period: str) -> tuple[pd.DataFrame, dict]:
    period_labels = history.set_index("period")["month_year_label"].to_dict()
    st.sidebar.markdown("## Filters")
    scoped = period_df.copy()

    circles = ["All"] + sorted(scoped["circle"].dropna().unique().tolist())
    selected_circle = st.sidebar.selectbox("Circle", circles)
    if selected_circle != "All":
        scoped = scoped[scoped["circle"] == selected_circle]

    divisions = ["All"] + sorted(scoped["division"].dropna().unique().tolist())
    selected_division = st.sidebar.selectbox("Division", divisions)
    if selected_division != "All":
        scoped = scoped[scoped["division"] == selected_division]

    subdivisions = ["All"] + sorted(scoped["subdivision"].dropna().unique().tolist())
    selected_subdivision = st.sidebar.selectbox("Subdivision", subdivisions)
    if selected_subdivision != "All":
        scoped = scoped[scoped["subdivision"] == selected_subdivision]

    analysis_focus = st.sidebar.selectbox(
        "Analysis Focus",
        [
            "All DTs",
            "Overloaded DTs",
            "Unbalanced DTs",
            "Both Overloaded and Unbalanced",
            "High Off-hours",
            "Near Overload DTs",
        ],
    )

    off_hours_threshold = st.sidebar.slider("High Off-hours Threshold", 0, 168, 24, 1)
    if analysis_focus == "Overloaded DTs":
        scoped = scoped[scoped["dt_loading_status"] == "high"]
    elif analysis_focus == "Unbalanced DTs":
        scoped = scoped[scoped["dt_unbalance_status"] == "high"]
    elif analysis_focus == "Both Overloaded and Unbalanced":
        scoped = scoped[
            (scoped["dt_loading_status"] == "high") & (scoped["dt_unbalance_status"] == "high")
        ]
    elif analysis_focus == "High Off-hours":
        scoped = scoped[scoped["power_off_hours"].fillna(0) >= off_hours_threshold]
    elif analysis_focus == "Near Overload DTs":
        scoped = scoped[scoped["dt_loading_status"] == "medium"]

    filter_info = {
        "selected_period": selected_period,
        "selected_period_label": period_labels.get(selected_period, selected_period),
        "selected_circle": selected_circle,
        "selected_division": selected_division,
        "selected_subdivision": selected_subdivision,
        "analysis_focus": analysis_focus,
        "off_hours_threshold": off_hours_threshold,
    }
    return scoped, filter_info


def build_summary_table(df: pd.DataFrame, off_hours_threshold: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["circle", "division", "subdivision"], dropna=False)
        .agg(
            total_dts=("dt_code", "count"),
            overloaded_dts=("dt_loading_status", lambda s: int((s == "high").sum())),
            near_overload_dts=("dt_loading_status", lambda s: int((s == "medium").sum())),
            unbalanced_dts=("dt_unbalance_status", lambda s: int((s == "high").sum())),
            high_off_hours_dts=("power_off_hours", lambda s: int((s.fillna(0) >= off_hours_threshold).sum())),
        )
        .reset_index()
        .sort_values(
            ["overloaded_dts", "unbalanced_dts", "high_off_hours_dts", "total_dts"],
            ascending=False,
        )
    )
    return summary


def style_summary_table(summary_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        summary_df.style.background_gradient(subset=["overloaded_dts"], cmap="OrRd")
        .background_gradient(subset=["unbalanced_dts"], cmap="YlGnBu")
        .background_gradient(subset=["high_off_hours_dts"], cmap="YlOrBr")
    )


def style_detail_table(detail_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        detail_df.style.format(
            {
                "dt_loading": "{:.1f}",
                "max_unbalance": "{:.3f}",
                "power_off_hours": "{:.1f}",
                "criticality_score": "{:.3f}",
                "predicted_failure_probability": "{:.2f}%",
            }
        )
        .background_gradient(subset=["dt_loading"], cmap="YlOrRd")
        .background_gradient(subset=["max_unbalance"], cmap="YlGnBu")
        .background_gradient(subset=["predicted_failure_probability"], cmap="RdYlGn_r")
        .map(
            lambda value: "font-weight: 800; color: #7a2412; background-color: #ffd8c7;"
            if pd.notna(value) and value >= 70
            else ("font-weight: 700; color: #8b5e00; background-color: #fff1bf;" if pd.notna(value) and value >= 45 else ""),
            subset=["predicted_failure_probability"],
        )
    )


def dataframe_download_bytes(df: pd.DataFrame) -> bytes:
    export_df = df.copy()
    if "predicted_failure_probability" in export_df.columns:
        export_df["predicted_failure_probability"] = (export_df["predicted_failure_probability"] * 100).round(2)
    return export_df.to_csv(index=False).encode("utf-8")


def render_metric_card(title: str, value: int, note: str, card_class: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            <h3>{title}</h3>
            <div class="metric-number">{value:,}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pick_prediction_chip(probability: float) -> tuple[str, str]:
    if probability >= 70:
        return "High Risk Signal", "chip-high"
    if probability >= 45:
        return "Moderate Risk Signal", "chip-medium"
    return "Lower Risk Signal", "chip-low"


def get_detail_selection(filtered_df: pd.DataFrame, threshold: int) -> tuple[str | None, pd.DataFrame]:
    detail_view = st.session_state.get("detail_view")
    if detail_view == "all":
        return "All Filtered DTs", filtered_df.copy()
    if detail_view == "overloaded":
        return "Overloaded DT Details", filtered_df[filtered_df["dt_loading_status"] == "high"].copy()
    if detail_view == "unbalanced":
        return "Unbalanced DT Details", filtered_df[filtered_df["dt_unbalance_status"] == "high"].copy()
    if detail_view == "off_hours":
        return "High Off-hours DT Details", filtered_df[filtered_df["power_off_hours"].fillna(0) >= threshold].copy()
    if detail_view == "near_overload":
        return "Near Overload DT Details", filtered_df[filtered_df["dt_loading_status"] == "medium"].copy()
    return None, pd.DataFrame()


def apply_summary_scope(df: pd.DataFrame) -> pd.DataFrame:
    scope = st.session_state.get("summary_scope")
    if not scope or df.empty:
        return df

    scoped = df.copy()
    for key in ["circle", "division", "subdivision"]:
        value = scope.get(key)
        if value is not None and key in scoped.columns:
            scoped = scoped[scoped[key] == value]
    return scoped


def extract_selected_rows(selection_event) -> list[int]:
    if selection_event is None:
        return []
    if isinstance(selection_event, dict):
        return selection_event.get("selection", {}).get("rows", []) or []

    selection = getattr(selection_event, "selection", None)
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    return rows or []


def scroll_to_detail_section() -> None:
    st.html(
        """
        <script>
        const tryScroll = () => {
          const el = window.parent.document.getElementById("detail-section");
          if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        };
        setTimeout(tryScroll, 80);
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )


def latest_period_per_calendar_month(history: pd.DataFrame) -> list[str]:
    temp = history.copy()
    temp["month_num"] = temp["period"].str[-2:].astype(int)
    latest = (
        temp.sort_values("period", ascending=False)
        .drop_duplicates(subset=["month_num"], keep="first")
        .sort_values("period", ascending=False)
    )
    return latest["period"].tolist()


def open_detail_from_summary(detail_view: str, row: pd.Series) -> None:
    st.session_state["detail_view"] = detail_view
    st.session_state["summary_scope"] = {
        "circle": row["circle"],
        "division": row["division"],
        "subdivision": row["subdivision"],
    }


def render_summary_button_grid(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    header = st.columns([1.4, 1.4, 1.7, 0.9, 0.9, 0.9, 0.9, 1.0])
    labels = ["Circle", "Division", "Subdivision", "Total", "Overloaded", "Near OL", "Unbalanced", "Off-hours"]
    for col, label in zip(header, labels):
        col.markdown(f"**{label}**")

    for idx, row in summary_df.iterrows():
        cols = st.columns([1.4, 1.4, 1.7, 0.9, 0.9, 0.9, 0.9, 1.0])
        cols[0].write(row["circle"])
        cols[1].write(row["division"])
        cols[2].write(row["subdivision"])
        if cols[3].button(f"{int(row['total_dts']):,}", key=f"sum_all_{idx}"):
            open_detail_from_summary("all", row)
        if cols[4].button(f"{int(row['overloaded_dts']):,}", key=f"sum_ol_{idx}", disabled=int(row["overloaded_dts"]) == 0):
            open_detail_from_summary("overloaded", row)
        if cols[5].button(f"{int(row['near_overload_dts']):,}", key=f"sum_near_{idx}", disabled=int(row["near_overload_dts"]) == 0):
            open_detail_from_summary("near_overload", row)
        if cols[6].button(f"{int(row['unbalanced_dts']):,}", key=f"sum_unb_{idx}", disabled=int(row["unbalanced_dts"]) == 0):
            open_detail_from_summary("unbalanced", row)
        if cols[7].button(f"{int(row['high_off_hours_dts']):,}", key=f"sum_off_{idx}", disabled=int(row["high_off_hours_dts"]) == 0):
            open_detail_from_summary("off_hours", row)


def main() -> None:
    inject_styles()

    with st.spinner("Preparing dashboard data for this deployment..."):
        ensure_database_ready()

    if "detail_view" not in st.session_state:
        st.session_state["detail_view"] = None
    if "summary_scope" not in st.session_state:
        st.session_state["summary_scope"] = None

    if st.sidebar.button("Refresh Dashboard Data"):
        st.cache_data.clear()
        st.session_state["detail_view"] = None
        st.session_state["summary_scope"] = None

    db_signature = get_db_signature()
    import_history = load_import_history_data(db_signature)
    if import_history.empty:
        st.warning("No data found in SQLite yet. Run `python import_monthly_data.py <excel-file>` first, then refresh this dashboard.")
        return

    period_labels = import_history.set_index("period")["month_year_label"].to_dict()
    periods = latest_period_per_calendar_month(import_history)
    default_period = periods[0]
    selected_period_from_state = st.session_state.get("selected_period", default_period)
    if selected_period_from_state not in set(periods):
        selected_period_from_state = default_period
    selected_period = st.sidebar.selectbox(
        "Month-Year",
        periods,
        index=periods.index(selected_period_from_state),
        format_func=lambda x: period_labels.get(x, x),
    )
    st.session_state["selected_period"] = selected_period
    period_df = load_period_dashboard_data(selected_period, db_signature)

    filtered_df, filter_info = build_filters(period_df, import_history, selected_period)
    off_hours_threshold = filter_info["off_hours_threshold"]
    selected_period_label = filter_info["selected_period_label"]
    prediction_target_label = (pd.to_datetime(f"{selected_period}-01") + pd.DateOffset(years=1)).strftime("%B %Y")

    month_rows = len(period_df)

    st.markdown(
        f"""
        <div class="hero">
            <h1>DT Health Command Center</h1>
            <p>Monthly transformer stress analytics with hierarchy-wise monitoring, clickable hotspot counts, and next-year same-month failure-risk highlighting.</p>
            <div class="badge-row">
                <span class="badge">Selected month: {selected_period_label}</span>
                <span class="badge">Target prediction month: {prediction_target_label}</span>
                <span class="badge">Rows in selected month: {month_rows:,}</span>
                <span class="badge">Focus: {filter_info["analysis_focus"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-tip">
            Choose month, geography, and focus first. Then click any count inside the summary grid to open the matching detail list and prediction panel for that circle/division/subdivision without page navigation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_count = len(filtered_df)
    overloaded_count = int((filtered_df["dt_loading_status"] == "high").sum())
    unbalanced_count = int((filtered_df["dt_unbalance_status"] == "high").sum())
    off_hours_count = int((filtered_df["power_off_hours"].fillna(0) >= off_hours_threshold).sum())

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Filtered DTs", total_count, "Summary total for the current month-year and filter selection.", "metric-total")
    with metric_cols[1]:
        render_metric_card("Overloaded DTs", overloaded_count, "Loading above 100% of KVA rating.", "metric-overload")
    with metric_cols[2]:
        render_metric_card("Unbalanced DTs", unbalanced_count, "Maximum phase unbalance above 0.30.", "metric-unbalance")
    with metric_cols[3]:
        render_metric_card("High Off-hours DTs", off_hours_count, f"Power off-hours greater than or equal to {off_hours_threshold}.", "metric-offhours")

    summary_df = build_summary_table(filtered_df, off_hours_threshold)
    st.markdown('<div class="section-card"><div class="section-title">Hierarchy Summary</div><div class="subtle">Circle, division, and subdivision level summary for the current filter selection.</div></div>', unsafe_allow_html=True)
    if summary_df.empty:
        st.warning("No records matched the current filters.")
    else:
        with st.container(height=420):
            render_summary_button_grid(summary_df)
        scope = st.session_state.get("summary_scope")
        if scope:
            st.success(
                f"Selected scope: {scope['circle']} / {scope['division']} / {scope['subdivision']}"
            )
            if st.button("Clear selected summary scope", key="clear_summary_scope"):
                st.session_state["summary_scope"] = None
                st.session_state["detail_view"] = None
                st.rerun()

    detail_title, detail_df = get_detail_selection(filtered_df, off_hours_threshold)
    if detail_title is None:
        st.info("Click any count link in the summary grid to reveal the detail list and prediction panel.")
    else:
        st.markdown('<div id="detail-section"></div>', unsafe_allow_html=True)
        scroll_to_detail_section()
        with st.spinner("Preparing prediction view for the selected month..."):
            prediction_scores_df, model_info = load_period_predictions(selected_period, db_signature)

        prediction_period_df = period_df.merge(
            prediction_scores_df,
            on=["period", "source_row_number"],
            how="left",
            validate="1:1",
        )

        prediction_filtered = prediction_period_df.copy()
        if filter_info["selected_circle"] != "All":
            prediction_filtered = prediction_filtered[prediction_filtered["circle"] == filter_info["selected_circle"]]
        if filter_info["selected_division"] != "All":
            prediction_filtered = prediction_filtered[prediction_filtered["division"] == filter_info["selected_division"]]
        if filter_info["selected_subdivision"] != "All":
            prediction_filtered = prediction_filtered[prediction_filtered["subdivision"] == filter_info["selected_subdivision"]]

        prediction_filtered = apply_summary_scope(prediction_filtered)

        detail_title, detail_df = get_detail_selection(prediction_filtered, off_hours_threshold)
        detail_df = detail_df.sort_values(
            ["predicted_failure_probability", "criticality_score", "dt_loading", "max_unbalance"],
            ascending=False,
        )
        detail_view_columns = [
            "circle",
            "division",
            "subdivision",
            "dt_code",
            "dt_name",
            "kva_rating",
            "maximum_kva",
            "dt_loading",
            "dt_loading_status",
            "max_unbalance",
            "dt_unbalance_status",
            "power_off_hours",
            "criticality_score",
            "prediction_target_period",
            "predicted_failure_probability",
            "prediction_method",
        ]
        display_df = detail_df[detail_view_columns].copy()
        if not display_df.empty:
            display_df["predicted_failure_probability"] = (display_df["predicted_failure_probability"] * 100).round(2)
            display_df["prediction_method"] = display_df["prediction_method"].map(
                lambda value: "Logistic Regression" if value == "model" else "Heuristic"
            )

        detail_left, detail_right = st.columns([1.85, 1.0])
        with detail_left:
            st.markdown(
                f'<div class="section-card"><div class="section-title">{detail_title}</div><div class="subtle">Prediction column is color-highlighted so high-risk rows stand out immediately.</div></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(style_detail_table(display_df), width="stretch", hide_index=True)
            st.download_button(
                label="Download current detail view as CSV",
                data=dataframe_download_bytes(detail_df[EXPORT_COLUMNS]),
                file_name=f"dt_dashboard_{selected_period}_{st.session_state['detail_view']}.csv",
                mime="text/csv",
            )

        with detail_right:
            if detail_df.empty:
                st.warning("No rows exist for the selected detail view.")
            else:
                avg_probability = float(detail_df["predicted_failure_probability"].mean() * 100)
                top_row = detail_df.iloc[0]
                chip_label, chip_class = pick_prediction_chip(avg_probability)
                validation_line = (
                    f"<p><b>Validation ROC-AUC:</b> {model_info['validation_auc']:.3f}</p>"
                    if model_info["validation_auc"] is not None
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="prediction-panel">
                        <div class="prediction-chip {chip_class}">{chip_label}</div>
                        <div class="section-title">Prediction Focus</div>
                        <p><b>Selected month:</b> {selected_period_label}</p>
                        <p><b>Prediction month:</b> {prediction_target_label}</p>
                        <p><b>Average failure probability:</b> <span style="font-size:1.45rem;font-weight:800;color:#9e3d1f;">{avg_probability:.2f}%</span></p>
                        <p><b>Highest-risk DT in this view:</b> {top_row['dt_code']} ({top_row['division']}/{top_row['subdivision']})</p>
                        <p><b>Top-row predicted probability:</b> <span style="font-size:1.2rem;font-weight:800;color:#7a2412;">{top_row['predicted_failure_probability'] * 100:.2f}%</span></p>
                        <p><b>Model:</b> {model_info["model_name"]}</p>
                        {validation_line}
                        <p><b>Training rows:</b> {model_info['training_rows']:,}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("Imported Periods and Data Basis"):
        st.dataframe(
            import_history[["period", "month_year_label", "row_count", "imported_at"]],
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
