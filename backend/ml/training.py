from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from backend.ml.config import (
    CLASSIFICATION_FEATURES,
    MODEL_DIR,
    RANDOM_STATE,
    REAL_MARKET_LOOKUP_PATH,
    REAL_MASTER_DATASET_PATH,
    REAL_PROFILE_DATASET_PATH,
    REAL_PRODUCTION_DATASET_PATH,
    REAL_RECOMMENDATION_DATASET_PATH,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
    REPORT_DIR,
    RANKING_POOL_SIZE,
)
from backend.ml.models import (
    create_catboost_classifier,
    create_lgbm_classifier,
    create_tabnet_classifier,
    create_yield_regressor,
    fit_tabnet_classifier,
    save_tabnet_model,
)

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    labels = np.arange(y_prob.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "top3_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=min(3, len(labels)), labels=labels)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _normalize_range(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {"min": 0.0, "max": 1.0}
    return {"min": float(clean.min()), "max": float(clean.max())}


def _to_float32_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _compute_shap_summary(model: Any, X_sample: np.ndarray, feature_names: list[str], report_dir: Path) -> dict[str, Any]:
    if shap is None:
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
            ranking = sorted(
                (
                    {"feature": feature_names[idx], "importance": float(score)}
                    for idx, score in enumerate(importance)
                ),
                key=lambda item: item["importance"],
                reverse=True,
            )
            summary = {
                "status": "fallback_feature_importance",
                "reason": "shap_not_installed",
                "top_features": ranking[:20],
            }
            with (report_dir / "shap_summary.json").open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)
            return summary
        return {"status": "skipped", "reason": "shap_not_installed"}

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        values = np.stack([np.abs(class_values) for class_values in shap_values], axis=0).mean(axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values = np.abs(shap_values).mean(axis=2)
    else:
        values = np.abs(np.asarray(shap_values))

    importance = values.mean(axis=0)
    ranking = sorted(
        ({"feature": feature_names[idx], "mean_abs_shap": float(score)} for idx, score in enumerate(importance)),
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )
    summary = {"status": "ok", "top_features": ranking[:20]}
    with (report_dir / "shap_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def _load_real_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommendation_df = pd.read_csv(REAL_RECOMMENDATION_DATASET_PATH)
    master_df = pd.read_csv(REAL_MASTER_DATASET_PATH)
    profile_df = pd.read_csv(REAL_PROFILE_DATASET_PATH)
    market_df = pd.read_csv(REAL_MARKET_LOOKUP_PATH)
    _ = pd.read_csv(REAL_PRODUCTION_DATASET_PATH)
    return recommendation_df, master_df, profile_df, market_df


def _classification_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), CLASSIFICATION_FEATURES)]
    )


def _regression_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), REGRESSION_NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                REGRESSION_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _group_signature(df: pd.DataFrame, columns: list[str], decimals: int = 3) -> np.ndarray:
    temp = df[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(temp[column]):
            temp[column] = temp[column].round(decimals)
        temp[column] = temp[column].astype(str)
    return temp.agg("|".join, axis=1).to_numpy()


def _cross_validated_classification(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_tabnet = np.zeros((len(X_raw), num_classes), dtype=np.float32)
    oof_lgbm = np.zeros((len(X_raw), num_classes), dtype=np.float32)
    oof_catboost = np.zeros((len(X_raw), num_classes), dtype=np.float32)

    for train_idx, valid_idx in splitter.split(X_raw, y, groups):
        preprocessor = _classification_preprocessor()
        X_train = _to_float32_array(preprocessor.fit_transform(X_raw.iloc[train_idx]))
        X_valid = _to_float32_array(preprocessor.transform(X_raw.iloc[valid_idx]))
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        tabnet_model = fit_tabnet_classifier(create_tabnet_classifier(num_classes), X_train, y_train, X_valid, y_valid)
        lgbm_model = create_lgbm_classifier(num_classes)
        lgbm_model.fit(X_train, y_train)
        catboost_model = create_catboost_classifier(num_classes)
        catboost_model.fit(X_train, y_train)

        oof_tabnet[valid_idx] = np.asarray(tabnet_model.predict_proba(X_valid), dtype=np.float32)
        oof_lgbm[valid_idx] = np.asarray(lgbm_model.predict_proba(X_valid), dtype=np.float32)
        oof_catboost[valid_idx] = np.asarray(catboost_model.predict_proba(X_valid), dtype=np.float32)

    cv_metrics = {
        "tabnet": _classification_metrics(y, np.argmax(oof_tabnet, axis=1), oof_tabnet),
        "lightgbm": _classification_metrics(y, np.argmax(oof_lgbm, axis=1), oof_lgbm),
        "catboost": _classification_metrics(y, np.argmax(oof_catboost, axis=1), oof_catboost),
    }
    return cv_metrics, oof_tabnet, oof_lgbm


def _cross_validated_regression(
    X_raw: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
) -> dict[str, float]:
    splitter = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(X_raw), dtype=np.float64)

    for train_idx, valid_idx in splitter.split(X_raw, y, groups):
        preprocessor = _regression_preprocessor()
        X_train = _to_float32_array(preprocessor.fit_transform(X_raw.iloc[train_idx]))
        X_valid = _to_float32_array(preprocessor.transform(X_raw.iloc[valid_idx]))
        y_train = y.iloc[train_idx]

        model = create_yield_regressor()
        model.fit(X_train, y_train)
        oof_pred[valid_idx] = model.predict(X_valid)

    return _regression_metrics(y.to_numpy(), oof_pred)


def train_and_save(
    dataset_path: Path | None = None,
    model_dir: Path | None = None,
    report_dir: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    del dataset_path
    del data_dir
    model_dir = Path(model_dir or MODEL_DIR)
    report_dir = Path(report_dir or REPORT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    recommendation_df, master_df, profile_df, market_df = _load_real_datasets()
    original_recommendation_rows = len(recommendation_df)
    original_master_rows = len(master_df)

    recommendation_df = recommendation_df.drop_duplicates(subset=CLASSIFICATION_FEATURES + ["crop"]).reset_index(drop=True)
    master_df = master_df.drop_duplicates(
        subset=REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES + ["yield"]
    ).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(recommendation_df["crop"])
    X_class = recommendation_df[CLASSIFICATION_FEATURES].copy()
    class_groups = _group_signature(X_class, CLASSIFICATION_FEATURES)
    num_classes = len(label_encoder.classes_)

    class_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(class_splitter.split(X_class, y_class, class_groups))
    X_train_raw = X_class.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_class.iloc[test_idx].reset_index(drop=True)
    y_train = y_class[train_idx]
    y_test = y_class[test_idx]
    class_groups_train = class_groups[train_idx]

    classification_cv_metrics, oof_tabnet, oof_lgbm = _cross_validated_classification(
        X_train_raw,
        y_train,
        class_groups_train,
        num_classes,
    )

    stacking_model = LogisticRegression(max_iter=2000)
    stacking_model.fit(np.hstack([oof_tabnet, oof_lgbm]), y_train)

    class_preprocessor = _classification_preprocessor()
    X_train = _to_float32_array(class_preprocessor.fit_transform(X_train_raw))
    X_test = _to_float32_array(class_preprocessor.transform(X_test_raw))
    class_feature_names = list(class_preprocessor.get_feature_names_out())

    tabnet_model = fit_tabnet_classifier(create_tabnet_classifier(num_classes), X_train, y_train, X_test, y_test)
    lgbm_model = create_lgbm_classifier(num_classes)
    lgbm_model.fit(X_train, y_train)
    catboost_model = create_catboost_classifier(num_classes)
    catboost_model.fit(X_train, y_train)

    tabnet_prob = np.asarray(tabnet_model.predict_proba(X_test))
    lgbm_prob = np.asarray(lgbm_model.predict_proba(X_test))
    catboost_prob = np.asarray(catboost_model.predict_proba(X_test))
    stacking_prob = np.asarray(stacking_model.predict_proba(np.hstack([tabnet_prob, lgbm_prob])))

    classification_metrics = {
        "tabnet": _classification_metrics(y_test, np.asarray(tabnet_model.predict(X_test)).astype(int), tabnet_prob),
        "lightgbm": _classification_metrics(y_test, lgbm_model.predict(X_test), lgbm_prob),
        "catboost": _classification_metrics(y_test, catboost_model.predict(X_test).reshape(-1).astype(int), catboost_prob),
        "stacking": _classification_metrics(y_test, stacking_model.predict(np.hstack([tabnet_prob, lgbm_prob])), stacking_prob),
    }

    regression_df = master_df.copy()
    regression_df["yield"] = regression_df["yield"].astype(float)
    X_reg = regression_df[REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES].copy()
    y_reg = regression_df["yield"].copy()
    reg_groups = _group_signature(regression_df, ["crop", "state_name", "district_name", "season"], decimals=0)

    reg_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    reg_train_idx, reg_test_idx = next(reg_splitter.split(X_reg, y_reg, reg_groups))
    X_reg_train_raw = X_reg.iloc[reg_train_idx].reset_index(drop=True)
    X_reg_test_raw = X_reg.iloc[reg_test_idx].reset_index(drop=True)
    y_reg_train = y_reg.iloc[reg_train_idx].reset_index(drop=True)
    y_reg_test = y_reg.iloc[reg_test_idx].reset_index(drop=True)

    regression_cv_metrics = _cross_validated_regression(
        X_reg_train_raw,
        y_reg_train,
        reg_groups[reg_train_idx],
    )

    regression_preprocessor = _regression_preprocessor()
    regression_preprocessor.fit(X_reg_train_raw)
    X_reg_train = _to_float32_array(regression_preprocessor.transform(X_reg_train_raw))
    X_reg_test = _to_float32_array(regression_preprocessor.transform(X_reg_test_raw))

    yield_model = create_yield_regressor()
    yield_model.fit(X_reg_train, y_reg_train)
    yield_pred = yield_model.predict(X_reg_test)
    regression_metrics = _regression_metrics(y_reg_test.to_numpy(), yield_pred)

    shap_summary = _compute_shap_summary(
        lgbm_model,
        X_test[: min(len(X_test), 300)],
        class_feature_names,
        report_dir,
    )

    tabnet_manifest = save_tabnet_model(tabnet_model, model_dir / "tabnet_model")
    joblib.dump(lgbm_model, model_dir / "lgbm_model.joblib")
    joblib.dump(catboost_model, model_dir / "catboost_model.joblib")
    joblib.dump(yield_model, model_dir / "yield_model.joblib")
    joblib.dump(stacking_model, model_dir / "stacking_model.joblib")
    joblib.dump(class_preprocessor, model_dir / "classification_preprocessor.joblib")
    joblib.dump(regression_preprocessor, model_dir / "regression_preprocessor.joblib")
    joblib.dump(label_encoder, model_dir / "crop_label_encoder.joblib")

    crop_profiles = {}
    market_by_crop = market_df.set_index("crop").to_dict(orient="index")
    master_yield = master_df.groupby("crop", as_index=False)["yield"].median()
    yield_map = dict(zip(master_yield["crop"], master_yield["yield"]))
    for row in profile_df.to_dict(orient="records"):
        crop = str(row["crop"])
        market_row = market_by_crop.get(crop, {})
        crop_profiles[crop] = {
            "ideal_nitrogen": float(row["ideal_nitrogen"]),
            "ideal_phosphorous": float(row["ideal_phosphorous"]),
            "ideal_potassium": float(row["ideal_potassium"]),
            "ideal_temperature_c": float(row["ideal_temperature_c"]),
            "ideal_humidity": float(row["ideal_humidity"]),
            "ideal_ph": float(row["ideal_ph"]),
            "ideal_rainfall_mm": float(row["ideal_rainfall_mm"]),
            "price_per_ton": float(market_row.get("price_per_ton", np.nan)) if pd.notna(market_row.get("price_per_ton", np.nan)) else None,
            "msp_per_ton": float(market_row.get("msp_per_ton", np.nan)) if pd.notna(market_row.get("msp_per_ton", np.nan)) else None,
            "historical_median_yield": float(yield_map.get(crop, np.nan)),
            "market_source": market_row.get("source_name"),
            "market_type": market_row.get("price_type"),
        }

    dataset_summary = {
        "recommendation_rows": int(len(recommendation_df)),
        "production_rows": int(len(master_df)),
        "crop_count": int(len(label_encoder.classes_)),
        "deduplication": {
            "recommendation_rows_removed": int(original_recommendation_rows - len(recommendation_df)),
            "production_rows_removed": int(original_master_rows - len(master_df)),
        },
        "evaluation_protocol": {
            "classification_holdout": "StratifiedGroupKFold split grouped by rounded classification feature signature",
            "classification_stacking": "Out-of-fold stacking on grouped training folds",
            "classification_cross_validation": "5-fold StratifiedGroupKFold on training partition",
            "regression_holdout": "GroupShuffleSplit grouped by crop+state+district+season",
            "regression_cross_validation": "5-fold GroupKFold on grouped training partition",
        },
        "season_options": sorted(master_df["season"].dropna().astype(str).unique().tolist()),
        "state_options": sorted(master_df["state_name"].dropna().astype(str).unique().tolist()),
        "district_options": sorted(master_df["district_name"].dropna().astype(str).unique().tolist()),
        "crop_year_range": {
            "min": int(master_df["crop_year"].min()),
            "max": int(master_df["crop_year"].max()),
        },
        "default_inputs": {
            "nitrogen": float(recommendation_df["nitrogen"].median()),
            "phosphorous": float(recommendation_df["phosphorous"].median()),
            "potassium": float(recommendation_df["potassium"].median()),
            "temperature_c": float(recommendation_df["temperature_c"].median()),
            "humidity": float(recommendation_df["humidity"].median()),
            "ph": float(recommendation_df["ph"].median()),
            "rainfall_mm": float(recommendation_df["rainfall_mm"].median()),
            "area": float(master_df["area"].median()),
            "crop_year": int(master_df["crop_year"].median()),
            "season": str(master_df["season"].mode().iloc[0]),
            "state_name": str(master_df["state_name"].mode().iloc[0]),
            "district_name": str(master_df["district_name"].mode().iloc[0]),
            "price_per_ton": float(market_df["price_per_ton"].dropna().median()),
        },
        "data_files": {
            "recommendation": str(REAL_RECOMMENDATION_DATASET_PATH),
            "production": str(REAL_PRODUCTION_DATASET_PATH),
            "profiles": str(REAL_PROFILE_DATASET_PATH),
            "market": str(REAL_MARKET_LOOKUP_PATH),
            "master": str(REAL_MASTER_DATASET_PATH),
        },
    }

    training_report = {
        "data_mode": "real_13_crop_grouped_holdout_pipeline",
        "dataset_summary": dataset_summary,
        "classification_metrics": classification_metrics,
        "classification_cv_metrics": classification_cv_metrics,
        "regression_metrics": regression_metrics,
        "regression_cv_metrics": regression_cv_metrics,
        "comparison": {
            "lightgbm_vs_catboost": {
                "accuracy_gap": round(classification_metrics["lightgbm"]["accuracy"] - classification_metrics["catboost"]["accuracy"], 6),
                "f1_gap": round(classification_metrics["lightgbm"]["f1_weighted"] - classification_metrics["catboost"]["f1_weighted"], 6),
            }
        },
        "shap_summary": shap_summary,
        "artifacts": {
            "tabnet_model": tabnet_manifest,
            "lgbm_model": "lgbm_model.joblib",
            "catboost_model": "catboost_model.joblib",
            "yield_model": "yield_model.joblib",
            "stacking_model": "stacking_model.joblib",
        },
    }

    runtime_metadata = {
        "dataset_summary": dataset_summary,
        "classification_features": CLASSIFICATION_FEATURES,
        "regression_numeric_features": REGRESSION_NUMERIC_FEATURES,
        "regression_categorical_features": REGRESSION_CATEGORICAL_FEATURES,
        "class_feature_names": class_feature_names,
        "crop_profiles": crop_profiles,
        "climate_ranges": {
            "temperature_c": _normalize_range(recommendation_df["temperature_c"]),
            "rainfall_mm": _normalize_range(recommendation_df["rainfall_mm"]),
            "humidity": _normalize_range(recommendation_df["humidity"]),
            "ph": _normalize_range(recommendation_df["ph"]),
        },
        "training_report": training_report,
        "ranking_pool_size": RANKING_POOL_SIZE,
    }

    with (model_dir / "runtime_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(runtime_metadata, file, indent=2)
    with (report_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(training_report, file, indent=2)

    return training_report


def main() -> None:
    report = train_and_save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


from backend.ml.training_strict import main as strict_main
from backend.ml.training_strict import train_and_save as strict_train_and_save

train_and_save = strict_train_and_save
main = strict_main
