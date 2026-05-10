from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ENTITY_DESCRIPTOR_COLUMNS = [
    "circle",
    "dt_code",
    "dt_meter_number",
    "dt_name",
    "kva_rating",
]

MAX_MODEL_TRAIN_ROWS = 12000


@dataclass
class PredictionResult:
    scored_df: pd.DataFrame
    method: str
    detail: str
    training_rows: int
    positive_rows: int
    validation_auc: float | None


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["period_date"] = pd.to_datetime(
        features["year"].astype(str) + "-" + features["month"].astype(str).str.zfill(2) + "-01"
    )
    features["month_name"] = features["period_date"].dt.strftime("%b")
    features = features.sort_values(["period_date"] + ENTITY_DESCRIPTOR_COLUMNS + ["source_row_number"]).reset_index(drop=True)
    features["dup_rank"] = features.groupby(["period"] + ENTITY_DESCRIPTOR_COLUMNS, dropna=False).cumcount() + 1
    features["entity_key"] = (
        features[ENTITY_DESCRIPTOR_COLUMNS].astype("string").fillna("").agg("|".join, axis=1)
        + "|"
        + features["dup_rank"].astype(str)
    )
    features = features.sort_values(["entity_key", "period_date"]).reset_index(drop=True)

    group = features.groupby("entity_key", group_keys=False)
    features["prev_dt_loading"] = group["dt_loading"].shift(1)
    features["prev_max_unbalance"] = group["max_unbalance"].shift(1)
    features["prev_power_off_hours"] = group["power_off_hours"].shift(1)

    features["rolling3_dt_loading_mean"] = (
        group["dt_loading"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    features["rolling3_unbalance_mean"] = (
        group["max_unbalance"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    features["rolling3_power_off_mean"] = (
        group["power_off_hours"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    features["history_row_count"] = group.cumcount()

    target_lookup = features[
        ["entity_key", "period_date", "loading_status_score", "unbalance_status_score", "power_off_hours"]
    ].rename(
        columns={
            "period_date": "target_period_date",
            "loading_status_score": "future_loading_score",
            "unbalance_status_score": "future_unbalance_score",
            "power_off_hours": "future_power_off_hours",
        }
    )

    features["target_period_date"] = features["period_date"] + pd.DateOffset(years=1)
    features = features.merge(target_lookup, on=["entity_key", "target_period_date"], how="left", validate="1:1")

    future_stress_score = (
        features["future_loading_score"].fillna(0) * 0.45
        + features["future_unbalance_score"].fillna(0) * 0.40
        + (features["future_power_off_hours"].fillna(0).clip(0, 72) / 72.0) * 0.15 * 3
    )
    features["future_failure_proxy"] = np.where(future_stress_score >= 1.6, 1, 0)
    features["future_failure_proxy"] = features["future_failure_proxy"].where(
        features["future_loading_score"].notna()
        | features["future_unbalance_score"].notna()
        | features["future_power_off_hours"].notna()
    )

    features["prediction_target_period"] = features["target_period_date"].dt.strftime("%Y-%m")
    return features


def _heuristic_probability(features: pd.DataFrame) -> np.ndarray:
    load_component = features["dt_loading"].fillna(0).clip(0, 180) / 180.0
    unbalance_component = features["max_unbalance"].fillna(0).clip(0, 0.6) / 0.6
    off_hours_component = features["power_off_hours"].fillna(0).clip(0, 72) / 72.0
    history_component = (
        features["rolling3_dt_loading_mean"].fillna(features["dt_loading"]).clip(0, 160) / 160.0
    )
    probability = (
        0.10
        + (0.38 * load_component)
        + (0.32 * unbalance_component)
        + (0.12 * off_hours_component)
        + (0.08 * history_component)
    )
    return np.clip(probability, 0.02, 0.98)


def _build_model_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def _downsample_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    if len(train_df) <= MAX_MODEL_TRAIN_ROWS:
        return train_df

    sample_fraction = MAX_MODEL_TRAIN_ROWS / float(len(train_df))
    sampled_parts: list[pd.DataFrame] = []
    for _, group in train_df.groupby(["target_period_date", "future_failure_proxy"], dropna=False):
        sample_size = max(1, int(round(len(group) * sample_fraction)))
        sampled_parts.append(group.sample(n=min(len(group), sample_size), random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) > MAX_MODEL_TRAIN_ROWS:
        sampled = sampled.sample(n=MAX_MODEL_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    return sampled


def attach_failure_predictions(
    df: pd.DataFrame,
    *,
    allow_train_without_holdout: bool = False,
    training_scope_label: str = "historical panel",
    score_period: str | None = None,
) -> PredictionResult:
    if df.empty:
        return PredictionResult(
            scored_df=df.copy(),
            method="unavailable",
            detail="No data found in SQLite.",
            training_rows=0,
            positive_rows=0,
            validation_auc=None,
        )

    features = _prepare_features(df)
    scored = (
        features[features["period"] == score_period].copy()
        if score_period is not None
        else features.copy()
    )

    numeric_features = [
        "kva_rating",
        "kwh",
        "kvah",
        "power_factor",
        "avg_kva",
        "maximum_kva",
        "r_phase_max_current",
        "y_phase_max_current",
        "b_phase_max_current",
        "unbalance_ry",
        "unbalance_yb",
        "unbalance_br",
        "max_unbalance",
        "dt_loading",
        "load_factor",
        "utilization_factor",
        "power_off_hours",
        "power_on_hours",
        "total_hours",
        "power_off_ratio",
        "criticality_score",
        "prev_dt_loading",
        "prev_max_unbalance",
        "prev_power_off_hours",
        "rolling3_dt_loading_mean",
        "rolling3_unbalance_mean",
        "rolling3_power_off_mean",
        "history_row_count",
    ]
    categorical_features = ["circle", "division"]

    heuristic_probability = _heuristic_probability(scored)
    labeled = features[features["future_failure_proxy"].notna()].copy()

    if labeled["future_failure_proxy"].nunique() < 2 or len(labeled) < 200:
        scored["predicted_failure_probability"] = heuristic_probability
        scored["prediction_method"] = "heuristic"
        return PredictionResult(
            scored_df=scored,
            method="heuristic",
            detail=(
                "Not enough same-month next-year history exists yet to train a supervised model. "
                "The dashboard is using a leakage-safe heuristic stress score instead."
            ),
            training_rows=int(len(labeled)),
            positive_rows=int(labeled["future_failure_proxy"].sum()) if not labeled.empty else 0,
            validation_auc=None,
        )

    model = _build_model_pipeline(numeric_features, categorical_features)
    labeled = labeled.sort_values("target_period_date")
    split_periods = labeled["target_period_date"].drop_duplicates().sort_values()
    validation_count = max(1, int(len(split_periods) * 0.2))
    validation_periods = set(split_periods.tail(validation_count))

    train_mask = ~labeled["target_period_date"].isin(validation_periods)
    train_df = labeled[train_mask]
    valid_df = labeled[~train_mask]

    if train_df["future_failure_proxy"].nunique() < 2 or valid_df.empty:
        if allow_train_without_holdout and labeled["future_failure_proxy"].nunique() >= 2:
            model.fit(labeled[numeric_features + categorical_features], labeled["future_failure_proxy"])
            scored["predicted_failure_probability"] = model.predict_proba(
                scored[numeric_features + categorical_features]
            )[:, 1]
            scored["prediction_method"] = "model"
            return PredictionResult(
                scored_df=scored,
                method="model",
                detail=(
                    f"Supervised model trained on all available {training_scope_label} records. "
                    "A separate temporal holdout ROC-AUC is not available yet because only one future same-month target period exists."
                ),
                training_rows=int(len(labeled)),
                positive_rows=int(labeled["future_failure_proxy"].sum()),
                validation_auc=None,
            )
        scored["predicted_failure_probability"] = heuristic_probability
        scored["prediction_method"] = "heuristic"
        return PredictionResult(
            scored_df=scored,
            method="heuristic",
            detail=(
                "Historical labels exist, but there are not enough time-separated periods for a proper holdout. "
                "The dashboard stayed on the fallback heuristic to avoid leakage-prone training."
            ),
            training_rows=int(len(labeled)),
            positive_rows=int(labeled["future_failure_proxy"].sum()),
            validation_auc=None,
        )

    fit_train_df = _downsample_training_frame(train_df)

    model.fit(fit_train_df[numeric_features + categorical_features], fit_train_df["future_failure_proxy"])
    scored["predicted_failure_probability"] = model.predict_proba(
        scored[numeric_features + categorical_features]
    )[:, 1]
    scored["prediction_method"] = "model"

    valid_probabilities = model.predict_proba(valid_df[numeric_features + categorical_features])[:, 1]
    validation_auc = roc_auc_score(valid_df["future_failure_proxy"], valid_probabilities)

    return PredictionResult(
        scored_df=scored,
        method="model",
        detail=(
            "Supervised next-year same-month proxy model trained with time-separated holdout periods. "
            "All lag features use only the current and earlier months to avoid data leakage."
        ),
        training_rows=int(len(fit_train_df)),
        positive_rows=int(fit_train_df["future_failure_proxy"].sum()),
        validation_auc=float(validation_auc),
    )


def attach_failure_predictions_for_period(df: pd.DataFrame, score_period: str) -> PredictionResult:
    same_month_result = attach_failure_predictions(
        df,
        allow_train_without_holdout=True,
        training_scope_label="same-calendar-month historical",
        score_period=score_period,
    )
    return PredictionResult(
        scored_df=same_month_result.scored_df,
        method=same_month_result.method,
        detail=(
            "Prediction uses only past available data from the same calendar month across years for next-year same-month forecasting. "
            + same_month_result.detail
        ),
        training_rows=same_month_result.training_rows,
        positive_rows=same_month_result.positive_rows,
        validation_auc=same_month_result.validation_auc,
    )
