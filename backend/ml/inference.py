from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.config import (
    CLASSIFICATION_FEATURES,
    MODEL_DIR,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
    TOP_K_DEFAULT,
)
from backend.ml.models import load_tabnet_model


@dataclass
class ModelBundle:
    classification_preprocessor: Any
    regression_preprocessor: Any
    label_encoder: Any
    tabnet_model: Any | None
    lgbm_model: Any
    catboost_model: Any
    yield_model: Any
    stacking_model: Any
    metadata: dict[str, Any]


def _normalize(value: float, minimum: float, maximum: float, invert: bool = False) -> float:
    if maximum <= minimum:
        score = 1.0
    else:
        score = (float(value) - minimum) / (maximum - minimum)
    score = min(max(score, 0.0), 1.0)
    return 1.0 - score if invert else score


def _candidate_scores(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _crop_economics(crop_name: str) -> dict[str, float]:
    crop_key = crop_name.strip().lower()
    orchard_crops = {"apple", "banana", "coconut", "coffee", "grapes", "mango", "orange", "papaya"}
    spacious_field_crops = {"jute", "maize", "rice"}
    pulse_crops = {"blackgram", "lentil"}

    if crop_key in orchard_crops:
        return {
            "cultivated_area_ratio": 0.72,
            "marketable_yield_ratio": 0.86,
            "profit_margin_ratio": 0.36,
        }
    if crop_key in spacious_field_crops:
        return {
            "cultivated_area_ratio": 0.84,
            "marketable_yield_ratio": 0.91,
            "profit_margin_ratio": 0.34,
        }
    if crop_key in pulse_crops:
        return {
            "cultivated_area_ratio": 0.82,
            "marketable_yield_ratio": 0.9,
            "profit_margin_ratio": 0.38,
        }
    return {
        "cultivated_area_ratio": 0.8,
        "marketable_yield_ratio": 0.89,
        "profit_margin_ratio": 0.35,
    }


def _to_float32_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_model_bundle(model_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(model_dir or MODEL_DIR)
    metadata = json.loads((model_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata.get("training_report", {}).get("artifacts", {})

    tabnet_model = None
    tabnet_meta = artifacts.get("tabnet_model")
    if tabnet_meta:
        tabnet_model = load_tabnet_model(model_dir / "tabnet_model", tabnet_meta["format"])
    elif (model_dir / "tabnet_model.joblib").exists():
        tabnet_model = load_tabnet_model(model_dir / "tabnet_model", "joblib")
    elif (model_dir / "tabnet_model.zip").exists():
        tabnet_model = load_tabnet_model(model_dir / "tabnet_model", "tabnet_zip")

    return ModelBundle(
        classification_preprocessor=joblib.load(model_dir / "classification_preprocessor.joblib"),
        regression_preprocessor=joblib.load(model_dir / "regression_preprocessor.joblib"),
        label_encoder=joblib.load(model_dir / "crop_label_encoder.joblib"),
        tabnet_model=tabnet_model,
        lgbm_model=joblib.load(model_dir / "lgbm_model.joblib"),
        catboost_model=joblib.load(model_dir / "catboost_model.joblib"),
        yield_model=joblib.load(model_dir / "yield_model.joblib"),
        stacking_model=joblib.load(model_dir / "stacking_model.joblib"),
        metadata=metadata,
    )


class InferenceEngine:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_artifacts(cls, model_dir: Path | None = None) -> "InferenceEngine":
        return cls(load_model_bundle(model_dir))

    def _fill_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.bundle.metadata["dataset_summary"]["default_inputs"]
        prepared = defaults.copy()
        prepared.update({k: v for k, v in payload.items() if v is not None})
        return prepared

    def _build_classification_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{feature: payload.get(feature) for feature in CLASSIFICATION_FEATURES}])

    def predict(self, payload: dict[str, Any], top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
        prepared = self._fill_defaults(payload)
        class_frame = self._build_classification_frame(prepared)

        X_class = _to_float32_array(self.bundle.classification_preprocessor.transform(class_frame))

        tabnet_probs = None
        if self.bundle.tabnet_model is not None:
            tabnet_probs = np.asarray(self.bundle.tabnet_model.predict_proba(X_class))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lgbm_probs = np.asarray(self.bundle.lgbm_model.predict_proba(X_class))

        data_mode = self.bundle.metadata.get("training_report", {}).get("data_mode")
        if data_mode == "strict_leakage_safe_pipeline":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                stacked_probs = np.asarray(self.bundle.stacking_model.predict_proba(X_class))[0]
        else:
            if tabnet_probs is None:
                raise RuntimeError("TabNet model is required for legacy inference artifacts but was not found.")
            stacked_probs = np.asarray(self.bundle.stacking_model.predict_proba(np.hstack([tabnet_probs, lgbm_probs])))[0]

        labels = self.bundle.label_encoder.inverse_transform(np.arange(len(stacked_probs)))
        ranked_indices = np.argsort(stacked_probs)[::-1][: self.bundle.metadata["ranking_pool_size"]]

        crop_profiles = self.bundle.metadata["crop_profiles"]
        climate_ranges = self.bundle.metadata["climate_ranges"]

        area = float(prepared["area"])
        temp = float(prepared["temperature_c"])
        rainfall = float(prepared["rainfall_mm"])
        humidity = float(prepared["humidity"])
        ph_value = float(prepared["ph"])
        nutrient_values = {
            "nitrogen": float(prepared["nitrogen"]),
            "phosphorous": float(prepared["phosphorous"]),
            "potassium": float(prepared["potassium"]),
        }

        candidates: list[dict[str, Any]] = []
        for idx in ranked_indices:
            crop_name = str(labels[idx])
            profile = crop_profiles.get(crop_name, {})

            yield_payload = {
                "crop": crop_name,
                "state_name": prepared["state_name"],
                "district_name": prepared["district_name"],
                "season": prepared["season"],
                "crop_year": prepared["crop_year"],
                "area": area,
                "ideal_nitrogen": profile.get("ideal_nitrogen"),
                "ideal_phosphorous": profile.get("ideal_phosphorous"),
                "ideal_potassium": profile.get("ideal_potassium"),
                "ideal_temperature_c": profile.get("ideal_temperature_c"),
                "ideal_humidity": profile.get("ideal_humidity"),
                "ideal_ph": profile.get("ideal_ph"),
                "ideal_rainfall_mm": profile.get("ideal_rainfall_mm"),
                "price_per_ton": prepared.get("price_per_ton") or profile.get("price_per_ton"),
            }
            reg_frame = pd.DataFrame(
                [
                    {
                        feature: yield_payload.get(feature)
                        for feature in REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES
                    }
                ]
            )
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                predicted_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            price_per_ton = float(yield_payload["price_per_ton"] or 0.0)
            economics = _crop_economics(crop_name)
            effective_area = area * economics["cultivated_area_ratio"]
            marketable_yield = max(predicted_yield, 0.0) * effective_area * economics["marketable_yield_ratio"]
            gross_revenue = marketable_yield * price_per_ton
            profit = gross_revenue * economics["profit_margin_ratio"]

            ideal_temp_value = profile.get("ideal_temperature_c")
            ideal_rain_value = profile.get("ideal_rainfall_mm")
            ideal_temp = float(temp if ideal_temp_value is None else ideal_temp_value)
            ideal_rain = float(rainfall if ideal_rain_value is None else ideal_rain_value)
            raw_risk = abs(temp - ideal_temp) + abs(rainfall - ideal_rain)

            temp_component = abs(temp - ideal_temp) / max(
                climate_ranges["temperature_c"]["max"] - climate_ranges["temperature_c"]["min"], 1.0
            )
            rain_component = abs(rainfall - ideal_rain) / max(
                climate_ranges["rainfall_mm"]["max"] - climate_ranges["rainfall_mm"]["min"], 1.0
            )
            normalized_risk = float(min(max((temp_component + rain_component) / 2.0, 0.0), 1.0))

            nutrient_gaps = [
                abs(nutrient_values["nitrogen"] - float(profile.get("ideal_nitrogen", nutrient_values["nitrogen"])))
                / max(float(profile.get("ideal_nitrogen", 1.0)) or 1.0, 1.0),
                abs(
                    nutrient_values["phosphorous"]
                    - float(profile.get("ideal_phosphorous", nutrient_values["phosphorous"]))
                )
                / max(float(profile.get("ideal_phosphorous", 1.0)) or 1.0, 1.0),
                abs(nutrient_values["potassium"] - float(profile.get("ideal_potassium", nutrient_values["potassium"])))
                / max(float(profile.get("ideal_potassium", 1.0)) or 1.0, 1.0),
                abs(ph_value - float(profile.get("ideal_ph", ph_value)))
                / max(climate_ranges["ph"]["max"] - climate_ranges["ph"]["min"], 1.0),
                abs(humidity - float(profile.get("ideal_humidity", humidity)))
                / max(climate_ranges["humidity"]["max"] - climate_ranges["humidity"]["min"], 1.0),
            ]
            sustainability = float(1.0 - min(max(np.mean(nutrient_gaps), 0.0), 1.0))

            candidates.append(
                {
                    "crop": crop_name,
                    "classification_probability": float(stacked_probs[idx]),
                    "tabnet_probability": None if tabnet_probs is None else float(tabnet_probs[0][idx]),
                    "lightgbm_probability": float(lgbm_probs[0][idx]),
                    "predicted_yield": float(predicted_yield),
                    "profit": float(profit),
                    "effective_cultivated_area_ha": float(effective_area),
                    "marketable_yield_tons": float(marketable_yield),
                    "gross_revenue": float(gross_revenue),
                    "cultivated_area_ratio": float(economics["cultivated_area_ratio"]),
                    "marketable_yield_ratio": float(economics["marketable_yield_ratio"]),
                    "profit_margin_ratio": float(economics["profit_margin_ratio"]),
                    "risk": float(normalized_risk),
                    "risk_score": float(1.0 - normalized_risk),
                    "sustainability_score": sustainability,
                    "market_price_per_ton": price_per_ton,
                    "market_source": profile.get("market_source"),
                    "market_type": profile.get("market_type"),
                    "historical_median_yield": profile.get("historical_median_yield"),
                    "raw_risk": float(raw_risk),
                }
            )

        yield_scores = _candidate_scores([item["predicted_yield"] for item in candidates])
        profit_scores = _candidate_scores([item["profit"] for item in candidates])

        for item, yield_score, profit_score in zip(candidates, yield_scores, profit_scores):
            item["yield_score"] = float(yield_score)
            item["profit_score"] = float(profit_score)
            item["final_score"] = float(
                0.4 * yield_score
                + 0.3 * profit_score
                + 0.2 * item["risk_score"]
                + 0.1 * item["sustainability_score"]
            )

        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        top_candidates = candidates[:top_k]

        return {
            "best_crop": top_candidates[0]["crop"],
            "top_crops": top_candidates,
            "training_summary": self.bundle.metadata["training_report"],
            "used_defaults": {
                key: prepared[key]
                for key in ["season", "state_name", "district_name", "crop_year", "price_per_ton"]
            },
        }
