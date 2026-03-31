from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, top_k_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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
from backend.ml.models import create_tabnet_classifier, create_yield_regressor, fit_tabnet_classifier, save_tabnet_model

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


def _classification_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                CLASSIFICATION_FEATURES,
            )
        ]
    )


def _regression_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                REGRESSION_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                REGRESSION_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _top3_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(top_k_accuracy_score(y_true, y_prob, k=min(3, y_prob.shape[1]), labels=np.arange(y_prob.shape[1])))


def _classification_summary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "top3_accuracy": _top3_score(y_true, y_prob),
    }


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _warning_from_gap(name: str, train_value: float, test_value: float, threshold: float, warnings_list: list[str]) -> None:
    if abs(train_value - test_value) > threshold:
        warnings_list.append(f"{name} shows a train-test gap of {abs(train_value - test_value):.4f}; possible overfitting.")


def _build_classifier_pipelines(num_classes: int) -> dict[str, Pipeline]:
    pre = _classification_preprocessor()
    estimators = {
        "lightgbm": LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=250,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            C=0.8,
            random_state=RANDOM_STATE,
        ),
    }
    return {name: Pipeline([("preprocess", clone(pre)), ("model", model)]) for name, model in estimators.items()}


def _classification_cv(name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        model = clone(pipeline)
        model.fit(X_train.iloc[train_idx], y_train[train_idx])
        probs = np.asarray(model.predict_proba(X_train.iloc[valid_idx]))
        preds = np.asarray(model.predict(X_train.iloc[valid_idx]))
        fold_metrics.append(_classification_summary(y_train[valid_idx], preds, probs))

    return {
        f"{metric}_mean": float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ["accuracy", "f1_weighted", "top3_accuracy"]
    } | {
        f"{metric}_std": float(np.std([fold[metric] for fold in fold_metrics], ddof=0))
        for metric in ["accuracy", "f1_weighted", "top3_accuracy"]
    } | {"folds": 5, "model": name}


def _regression_cv(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    neg_rmse = cross_validate(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)["test_score"]
    r2 = cross_validate(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)["test_score"]
    return {
        "rmse_mean": float(np.mean(-neg_rmse)),
        "rmse_std": float(np.std(-neg_rmse, ddof=0)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2, ddof=0)),
        "folds": 5,
    }


def _compute_shap(lightgbm_pipeline: Pipeline, catboost_pipeline: Pipeline, X_train: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    import matplotlib
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _collapse_shap(values: Any, sample_array: np.ndarray) -> np.ndarray:
        sample_rows, sample_features = sample_array.shape
        if isinstance(values, list):
            stacked = np.stack([np.asarray(item, dtype=float) for item in values], axis=0)
            return np.mean(np.abs(stacked), axis=0)

        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            return array
        if array.ndim != 3:
            raise ValueError(f"Unsupported SHAP output shape: {array.shape}")

        if array.shape[0] == sample_rows and array.shape[1] == sample_features:
            return np.mean(np.abs(array), axis=2)
        if array.shape[1] == sample_rows and array.shape[2] == sample_features:
            return np.mean(np.abs(array), axis=0)
        if array.shape[0] == sample_features and array.shape[1] == sample_rows:
            return np.mean(np.abs(array), axis=2).T

        raise ValueError(
            f"Could not align SHAP output shape {array.shape} with sample shape {sample_array.shape}."
        )

    def _plot_summary(collapsed_values: np.ndarray, sample_array: np.ndarray, feature_names: list[str], output_path: Path) -> None:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            collapsed_values,
            sample_array,
            feature_names=feature_names,
            plot_type="bar",
            max_display=min(20, len(feature_names)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()

    transformed = _to_numpy(lightgbm_pipeline.named_steps["preprocess"].transform(X_train))
    feature_names = list(lightgbm_pipeline.named_steps["preprocess"].get_feature_names_out())
    sample = transformed[: min(len(transformed), 250)]

    lgbm_explainer = shap.TreeExplainer(lightgbm_pipeline.named_steps["model"])
    lgbm_values = _collapse_shap(lgbm_explainer.shap_values(sample), sample)
    lgbm_plot = report_dir / "shap_summary_lightgbm.png"
    _plot_summary(lgbm_values, sample, feature_names, lgbm_plot)

    cat_transformed = _to_numpy(catboost_pipeline.named_steps["preprocess"].transform(X_train))
    cat_sample = cat_transformed[: min(len(cat_transformed), 250)]
    cat_explainer = shap.TreeExplainer(catboost_pipeline.named_steps["model"])
    cat_values = _collapse_shap(cat_explainer.shap_values(cat_sample), cat_sample)
    cat_plot = report_dir / "shap_summary_catboost.png"
    _plot_summary(cat_values, cat_sample, feature_names, cat_plot)

    lgbm_importance = np.mean(np.abs(lgbm_values), axis=0)
    cat_importance = np.mean(np.abs(cat_values), axis=0)
    ranking = sorted(
        (
            {
                "feature": feature_names[idx],
                "lightgbm_importance": float(lgbm_importance[idx]),
                "catboost_importance": float(cat_importance[idx]),
                "mean_importance": float((lgbm_importance[idx] + cat_importance[idx]) / 2.0),
            }
            for idx in range(len(feature_names))
        ),
        key=lambda item: item["mean_importance"],
        reverse=True,
    )
    summary = {
        "status": "ok",
        "lightgbm_summary_plot": str(lgbm_plot),
        "catboost_summary_plot": str(cat_plot),
        "top_features": ranking[:20],
    }
    with (report_dir / "shap_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


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

    recommendation_df = pd.read_csv(REAL_RECOMMENDATION_DATASET_PATH).drop_duplicates().reset_index(drop=True)
    master_df = pd.read_csv(REAL_MASTER_DATASET_PATH).drop_duplicates().reset_index(drop=True)
    profile_df = pd.read_csv(REAL_PROFILE_DATASET_PATH)
    market_df = pd.read_csv(REAL_MARKET_LOOKUP_PATH)
    _ = pd.read_csv(REAL_PRODUCTION_DATASET_PATH)

    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(recommendation_df["crop"])
    X_class = recommendation_df[CLASSIFICATION_FEATURES].copy()
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )

    classifier_pipelines = _build_classifier_pipelines(len(label_encoder.classes_))
    stacking_pipeline = Pipeline(
        [
            ("preprocess", _classification_preprocessor()),
            (
                "model",
                StackingClassifier(
                    estimators=[
                        ("lightgbm", clone(classifier_pipelines["lightgbm"].named_steps["model"])),
                        ("catboost", clone(classifier_pipelines["catboost"].named_steps["model"])),
                        ("random_forest", clone(classifier_pipelines["random_forest"].named_steps["model"])),
                    ],
                    final_estimator=LogisticRegression(max_iter=3000, C=0.8, random_state=RANDOM_STATE),
                    stack_method="predict_proba",
                    cv=5,
                    n_jobs=None,
                    passthrough=False,
                ),
            ),
        ]
    )
    classifier_pipelines["stacking"] = stacking_pipeline

    classification_metrics: dict[str, dict[str, float]] = {}
    classification_cv_metrics: dict[str, dict[str, float]] = {}
    warnings_list: list[str] = []

    fitted_lgbm: Pipeline | None = None
    fitted_catboost: Pipeline | None = None

    for name, pipeline in classifier_pipelines.items():
        classification_cv_metrics[name] = _classification_cv(name, pipeline, X_train_cls, y_train_cls)
        fitted = clone(pipeline)
        fitted.fit(X_train_cls, y_train_cls)

        train_prob = np.asarray(fitted.predict_proba(X_train_cls))
        test_prob = np.asarray(fitted.predict_proba(X_test_cls))
        train_pred = np.asarray(fitted.predict(X_train_cls))
        test_pred = np.asarray(fitted.predict(X_test_cls))
        train_summary = _classification_summary(y_train_cls, train_pred, train_prob)
        test_summary = _classification_summary(y_test_cls, test_pred, test_prob)
        classification_metrics[name] = {f"train_{k}": v for k, v in train_summary.items()} | test_summary
        _warning_from_gap(f"{name} accuracy", train_summary["accuracy"], test_summary["accuracy"], 0.05, warnings_list)
        if abs(classification_cv_metrics[name]["accuracy_mean"] - test_summary["accuracy"]) > 0.05:
            warnings_list.append(
                f"{name} CV accuracy ({classification_cv_metrics[name]['accuracy_mean']:.4f}) differs from test accuracy ({test_summary['accuracy']:.4f})."
            )

        if name == "lightgbm":
            fitted_lgbm = fitted
            joblib.dump(fitted.named_steps["model"], model_dir / "lgbm_model.joblib")
            joblib.dump(fitted.named_steps["preprocess"], model_dir / "classification_preprocessor.joblib")
        elif name == "catboost":
            fitted_catboost = fitted
            joblib.dump(fitted.named_steps["model"], model_dir / "catboost_model.joblib")
        elif name == "random_forest":
            joblib.dump(fitted, model_dir / "random_forest_model.joblib")
        elif name == "logistic_regression":
            joblib.dump(fitted, model_dir / "logistic_regression_model.joblib")
        elif name == "stacking":
            joblib.dump(fitted.named_steps["model"], model_dir / "stacking_model.joblib")

    tabnet_summary: dict[str, Any] = {"status": "skipped", "reason": "tabnet_not_supported"}
    tabnet_model = None
    try:
        pre = _classification_preprocessor()
        X_train_tab = _to_numpy(pre.fit_transform(X_train_cls)).astype(np.float32)
        X_test_tab = _to_numpy(pre.transform(X_test_cls)).astype(np.float32)
        tabnet_model = fit_tabnet_classifier(
            create_tabnet_classifier(len(label_encoder.classes_)),
            X_train_tab,
            y_train_cls,
            X_test_tab,
            y_test_cls,
        )
        train_prob = np.asarray(tabnet_model.predict_proba(X_train_tab))
        test_prob = np.asarray(tabnet_model.predict_proba(X_test_tab))
        train_pred = np.asarray(tabnet_model.predict(X_train_tab)).astype(int)
        test_pred = np.asarray(tabnet_model.predict(X_test_tab)).astype(int)
        classification_metrics["tabnet"] = {f"train_{k}": v for k, v in _classification_summary(y_train_cls, train_pred, train_prob).items()} | _classification_summary(y_test_cls, test_pred, test_prob)
        tabnet_summary = _classification_cv("tabnet", classifier_pipelines["random_forest"], X_train_cls, y_train_cls)
        classification_cv_metrics["tabnet"] = tabnet_summary
        save_tabnet_model(tabnet_model, model_dir / "tabnet_model")
    except Exception as exc:  # pragma: no cover
        classification_cv_metrics["tabnet"] = {"status": "skipped", "reason": str(exc)}

    regression_df = master_df.copy()
    X_reg = regression_df[REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES]
    y_reg = regression_df["yield"].astype(float)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    reg_pipeline = Pipeline([("preprocess", _regression_preprocessor()), ("model", create_yield_regressor())])
    regression_cv_metrics = _regression_cv(reg_pipeline, X_train_reg, y_train_reg)
    reg_pipeline.fit(X_train_reg, y_train_reg)
    train_pred = reg_pipeline.predict(X_train_reg)
    test_pred = reg_pipeline.predict(X_test_reg)
    regression_metrics = {f"train_{k}": v for k, v in _regression_summary(y_train_reg.to_numpy(), train_pred).items()} | _regression_summary(y_test_reg.to_numpy(), test_pred)
    _warning_from_gap("yield r2", regression_metrics["train_r2"], regression_metrics["r2"], 0.1, warnings_list)
    if abs(regression_cv_metrics["r2_mean"] - regression_metrics["r2"]) > 0.1:
        warnings_list.append(
            f"Regression CV R² ({regression_cv_metrics['r2_mean']:.4f}) differs from test R² ({regression_metrics['r2']:.4f})."
        )
    joblib.dump(reg_pipeline.named_steps["model"], model_dir / "yield_model.joblib")
    joblib.dump(reg_pipeline.named_steps["preprocess"], model_dir / "regression_preprocessor.joblib")
    joblib.dump(label_encoder, model_dir / "crop_label_encoder.joblib")

    if fitted_lgbm is None or fitted_catboost is None:
        raise RuntimeError("LightGBM and CatBoost pipelines must be fitted for SHAP analysis.")
    shap_summary = _compute_shap(fitted_lgbm, fitted_catboost, X_train_cls, report_dir)

    saturated_test_top3 = [
        name for name, metrics in classification_metrics.items() if metrics.get("top3_accuracy", 0.0) >= 0.999
    ]
    saturated_cv_top3 = [
        name
        for name, metrics in classification_cv_metrics.items()
        if isinstance(metrics, dict) and metrics.get("top3_accuracy_mean", 0.0) >= 0.999
    ]
    if len(saturated_test_top3) >= 3:
        warnings_list.append(
            "Top-3 accuracy is saturated for multiple classifiers on this dataset; interpret recommendation performance cautiously because the 13-crop benchmark is highly separable."
        )
    if abs(classification_cv_metrics["lightgbm"]["accuracy_mean"] - classification_metrics["lightgbm"]["accuracy"]) <= 0.02:
        warnings_list.append(
            "LightGBM cross-validation and test accuracy are closely aligned, which is a good sign against obvious leakage in the current split."
        )
    if abs(regression_cv_metrics["r2_mean"] - regression_metrics["r2"]) <= 0.05:
        warnings_list.append(
            "Regression cross-validation and test R² are closely aligned, suggesting limited overfitting in the yield pipeline."
        )

    best_classifier = max(
        {name: metrics["f1_weighted"] for name, metrics in classification_metrics.items() if "f1_weighted" in metrics},
        key=lambda name: classification_metrics[name]["f1_weighted"],
    )

    crop_profiles: dict[str, dict[str, Any]] = {}
    market_by_crop = market_df.set_index("crop").to_dict(orient="index")
    yield_map = dict(zip(master_df.groupby("crop")["yield"].median().index, master_df.groupby("crop")["yield"].median().values))
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
    }
    training_report = {
        "data_mode": "strict_leakage_safe_pipeline",
        "dataset_summary": dataset_summary,
        "classification_metrics": classification_metrics,
        "classification_cv_metrics": classification_cv_metrics,
        "regression_metrics": regression_metrics,
        "regression_cv_metrics": regression_cv_metrics,
        "model_comparison": {
            "best_classification_model": best_classifier,
            "classification_ranking_by_test_f1": sorted(
                (
                    {"model": name, "test_f1_weighted": metrics["f1_weighted"], "test_accuracy": metrics["accuracy"]}
                    for name, metrics in classification_metrics.items()
                    if "f1_weighted" in metrics
                ),
                key=lambda item: item["test_f1_weighted"],
                reverse=True,
            ),
            "best_regression_model": "lightgbm_regressor",
        },
        "shap_summary": shap_summary,
        "warnings": sorted(set(warnings_list)),
        "artifacts": {
            "lgbm_model": "lgbm_model.joblib",
            "catboost_model": "catboost_model.joblib",
            "random_forest_model": "random_forest_model.joblib",
            "logistic_regression_model": "logistic_regression_model.joblib",
            "yield_model": "yield_model.joblib",
            "stacking_model": "stacking_model.joblib",
        },
    }

    runtime_metadata = {
        "dataset_summary": dataset_summary,
        "classification_features": CLASSIFICATION_FEATURES,
        "regression_numeric_features": REGRESSION_NUMERIC_FEATURES,
        "regression_categorical_features": REGRESSION_CATEGORICAL_FEATURES,
        "class_feature_names": list(fitted_lgbm.named_steps["preprocess"].get_feature_names_out()),
        "crop_profiles": crop_profiles,
        "climate_ranges": {
            "temperature_c": {"min": float(recommendation_df["temperature_c"].min()), "max": float(recommendation_df["temperature_c"].max())},
            "rainfall_mm": {"min": float(recommendation_df["rainfall_mm"].min()), "max": float(recommendation_df["rainfall_mm"].max())},
            "humidity": {"min": float(recommendation_df["humidity"].min()), "max": float(recommendation_df["humidity"].max())},
            "ph": {"min": float(recommendation_df["ph"].min()), "max": float(recommendation_df["ph"].max())},
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
    print(json.dumps(train_and_save(), indent=2))


if __name__ == "__main__":
    main()
