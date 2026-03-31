from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.config_realistic_v2 import (
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CROP_PROFILES_DATASET_PATH,
    MARKET_PRICES_DATASET_PATH,
    MODEL_DIR,
    PROJECT_MASTER_DATASET_PATH,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
)
from backend.ml.crop_rules_realistic_v2 import (
    CROP_DATABASE,
    current_analysis_context,
    derive_season,
    harvest_month,
    late_sowing_penalty,
    month_name,
    water_need_factor,
)

TOP_K_DEFAULT = 3
RANKING_POOL_SIZE = 13


@dataclass
class ModelBundle:
    classification_preprocessor: Any
    regression_preprocessor: Any
    label_encoder: Any
    lgbm_model: Any
    catboost_model: Any
    yield_model: Any
    stacking_model: Any
    metadata: dict[str, Any]


def _to_float32_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _candidate_scores(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


def _build_metadata(runtime_metadata: dict[str, Any], master_df: pd.DataFrame, profiles_df: pd.DataFrame, market_df: pd.DataFrame) -> dict[str, Any]:
    defaults = {
        "nitrogen": float(master_df["nitrogen"].median()),
        "phosphorous": float(master_df["phosphorous"].median()),
        "potassium": float(master_df["potassium"].median()),
        "temperature_c": float(master_df["temperature_c"].median()),
        "humidity": float(master_df["humidity"].median()),
        "ph": float(master_df["ph"].median()),
        "rainfall_mm": float(master_df["rainfall_mm"].median()),
        "moisture": float(master_df["moisture"].median()),
        "area": float(master_df["area_ha"].median()),
        "season": str(master_df["season"].mode().iloc[0]),
        "state_name": str(master_df["state_name"].mode().iloc[0]),
        "district_name": str(master_df["district_name"].mode().iloc[0]),
        "crop_year": int(master_df["crop_year"].median()),
        "price_per_ton": float(master_df["price_per_ton"].median()),
    }

    climate_ranges = {
        "temperature_c": {"min": float(master_df["temperature_c"].min()), "max": float(master_df["temperature_c"].max())},
        "rainfall_mm": {"min": float(master_df["rainfall_mm"].min()), "max": float(master_df["rainfall_mm"].max())},
        "humidity": {"min": float(master_df["humidity"].min()), "max": float(master_df["humidity"].max())},
        "ph": {"min": float(master_df["ph"].min()), "max": float(master_df["ph"].max())},
    }

    crop_profiles: dict[str, dict[str, Any]] = {}
    for crop_name, frame in profiles_df.groupby("crop", sort=False):
        typical = frame.loc[frame["profile_variant"] == "typical"]
        row = typical.iloc[0] if not typical.empty else frame.iloc[0]
        crop_market = market_df.loc[market_df["crop"] == crop_name, "price_per_ton"]
        crop_yield = master_df.loc[master_df["crop"] == crop_name, "target_yield_t_ha"]
        crop_key = str(crop_name).strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        crop_profiles[str(crop_name)] = {
            "min_nitrogen": float(row["min_nitrogen"]),
            "max_nitrogen": float(row["max_nitrogen"]),
            "min_phosphorous": float(row["min_phosphorous"]),
            "max_phosphorous": float(row["max_phosphorous"]),
            "min_potassium": float(row["min_potassium"]),
            "max_potassium": float(row["max_potassium"]),
            "min_temp_c": float(row["min_temp_c"]),
            "max_temp_c": float(row["max_temp_c"]),
            "min_humidity": float(row["min_humidity"]),
            "max_humidity": float(row["max_humidity"]),
            "min_ph": float(row["min_ph"]),
            "max_ph": float(row["max_ph"]),
            "min_rainfall_mm": float(row["min_rainfall_mm"]),
            "max_rainfall_mm": float(row["max_rainfall_mm"]),
            "price_per_ton": float(crop_market.median()) if not crop_market.empty else defaults["price_per_ton"],
            "market_source": "market_prices_realistic.csv",
            "market_type": "state_season_realistic_market",
            "historical_median_yield": float(crop_yield.median()) if not crop_yield.empty else None,
            "duration_months": int(crop_rules.get("duration_months", 6)),
            "sowing_months": list(crop_rules.get("sowing_months", [6, 7, 8])),
            "peak_harvest_months": list(crop_rules.get("peak_harvest_months", [10, 11])),
        }

    return {
        "dataset_summary": {
            "recommendation_rows": int(runtime_metadata["training_report"]["dataset_summary"]["classification_rows"]),
            "production_rows": int(runtime_metadata["training_report"]["dataset_summary"]["regression_rows"]),
            "crop_count": int(master_df["crop"].nunique()),
            "default_inputs": defaults,
        },
        "training_report": runtime_metadata["training_report"],
        "crop_profiles": crop_profiles,
        "climate_ranges": climate_ranges,
        "ranking_pool_size": RANKING_POOL_SIZE,
        "market_prices": market_df.to_dict(orient="records"),
    }


def load_model_bundle(model_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(model_dir or MODEL_DIR)
    runtime_metadata = json.loads((model_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    master_df = pd.read_csv(PROJECT_MASTER_DATASET_PATH)
    profiles_df = pd.read_csv(CROP_PROFILES_DATASET_PATH)
    market_df = pd.read_csv(MARKET_PRICES_DATASET_PATH)
    metadata = _build_metadata(runtime_metadata, master_df, profiles_df, market_df)

    return ModelBundle(
        classification_preprocessor=joblib.load(model_dir / "classification_preprocessor.joblib"),
        regression_preprocessor=joblib.load(model_dir / "regression_preprocessor.joblib"),
        label_encoder=joblib.load(model_dir / "crop_label_encoder.joblib"),
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
        prepared["season"] = derive_season(date.today().month)
        prepared["crop_year"] = date.today().year
        return prepared

    def _market_price_rs_per_kg(self, crop_name: str, state_name: str, season: str) -> float:
        market_df = pd.DataFrame(self.bundle.metadata["market_prices"])
        crop_key = crop_name.strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        minimum, maximum = crop_rules.get("price_range_rs_per_kg", (12.0, 40.0))
        baseline = crop_rules.get("base_price_rs_per_kg", (minimum + maximum) / 2.0)
        exact = market_df[
            (market_df["crop"] == crop_name)
            & (market_df["state_name"] == state_name)
            & (market_df["season"].astype(str).str.strip() == season)
        ]
        if not exact.empty:
            observed = float(exact["price_per_ton"].median()) / 1000.0
        else:
            crop_only = market_df[market_df["crop"] == crop_name]
            if not crop_only.empty:
                observed = float(crop_only["price_per_ton"].median()) / 1000.0
            else:
                observed = float(baseline)
        return _clip((0.55 * float(baseline)) + (0.45 * observed), float(minimum), float(maximum))

    def _build_regression_frame(self, prepared: dict[str, Any], crop_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "area_ha": float(prepared["area"]),
                    "nitrogen": float(prepared["nitrogen"]),
                    "phosphorous": float(prepared["phosphorous"]),
                    "potassium": float(prepared["potassium"]),
                    "temperature_c": float(prepared["temperature_c"]),
                    "humidity": float(prepared["humidity"]),
                    "ph": float(prepared["ph"]),
                    "rainfall_mm": float(prepared["rainfall_mm"]),
                    "moisture": float(prepared["moisture"]),
                    "crop": crop_name,
                    "state_name": prepared["state_name"],
                    "district_name": prepared["district_name"],
                    "season": prepared["season"],
                }
            ]
        )

    def _baseline_yield(self, crop_name: str, predicted_yield: float, profile: dict[str, Any], crop_rules: dict[str, Any]) -> float:
        yield_candidates = [float(predicted_yield), float(crop_rules["yield_avg_t_ha"])]
        if profile.get("historical_median_yield") is not None:
            yield_candidates.append(float(profile["historical_median_yield"]))
        return max(float(np.median(yield_candidates)), 0.15)

    def _climate_adjustments(self, prepared: dict[str, Any], profile: dict[str, Any], crop_rules: dict[str, Any]) -> tuple[float, bool, list[str]]:
        penalties: list[float] = []
        reasons: list[str] = []

        rainfall = float(prepared["rainfall_mm"])
        rain_min = float(profile["min_rainfall_mm"])
        rain_max = float(profile["max_rainfall_mm"])
        if rainfall < rain_min:
            deficit = (rain_min - rainfall) / max(rain_min, 1.0)
            penalties.append(0.10 if deficit <= 0.30 else 0.22)
            reasons.append("rainfall deficit")
            if deficit > 0.55:
                return 1.0, True, reasons + ["severe rainfall mismatch"]
        elif rainfall > rain_max:
            excess = (rainfall - rain_max) / max(rain_max, 1.0)
            penalties.append(0.08 if excess <= 0.30 else 0.16)
            reasons.append("rainfall excess")
            if excess > 0.70 and crop_rules["water_need"] == "low":
                return 1.0, True, reasons + ["severe rainfall excess"]

        temperature = float(prepared["temperature_c"])
        temp_min = float(profile["min_temp_c"])
        temp_max = float(profile["max_temp_c"])
        temp_span = max(temp_max - temp_min, 1.0)
        if temperature < temp_min:
            mismatch = (temp_min - temperature) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature below range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]
        elif temperature > temp_max:
            mismatch = (temperature - temp_max) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature above range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]

        humidity = float(prepared["humidity"])
        if humidity < float(profile["min_humidity"]) or humidity > float(profile["max_humidity"]):
            penalties.append(0.03)
            reasons.append("humidity mismatch")

        ph_value = float(prepared["ph"])
        ph_min = float(profile["min_ph"])
        ph_max = float(profile["max_ph"])
        if ph_value < ph_min or ph_value > ph_max:
            ph_gap = min(abs(ph_value - ph_min), abs(ph_value - ph_max))
            penalties.append(0.05 if ph_gap <= 0.6 else 0.10)
            reasons.append("soil pH mismatch")
            if ph_gap > 1.2:
                return 1.0, True, reasons + ["severe soil pH mismatch"]

        nutrient_penalty = 0.0
        for key, min_key, max_key in [
            ("nitrogen", "min_nitrogen", "max_nitrogen"),
            ("phosphorous", "min_phosphorous", "max_phosphorous"),
            ("potassium", "min_potassium", "max_potassium"),
        ]:
            value = float(prepared[key])
            if value < float(profile[min_key]) or value > float(profile[max_key]):
                nutrient_penalty += 0.02
        if nutrient_penalty:
            penalties.append(min(nutrient_penalty, 0.06))
            reasons.append("nutrient mismatch")

        moisture = float(prepared["moisture"])
        if crop_rules["water_need"] == "high" and moisture < 28:
            penalties.append(0.06)
            reasons.append("low field moisture")
        elif crop_rules["water_need"] == "low" and moisture > 55:
            penalties.append(0.03)
            reasons.append("excess field moisture")

        return min(sum(penalties), 0.70), False, reasons

    def _price_adjustment(self, crop_name: str, sowing_month: int, crop_rules: dict[str, Any], prepared: dict[str, Any]) -> tuple[float, str]:
        base_price = self._market_price_rs_per_kg(crop_name, str(prepared["state_name"]), str(prepared["season"]))
        harvest_month_number = harvest_month(sowing_month, int(crop_rules["duration_months"]))
        peak_months = set(crop_rules["peak_harvest_months"])
        if harvest_month_number in peak_months:
            factor = 0.82 if float(crop_rules["price_volatility"]) >= 0.25 else 0.88
            reason = "peak harvest supply"
        elif float(crop_rules["price_volatility"]) >= 0.26:
            factor = 1.12
            reason = "off-season support"
        else:
            factor = 1.03
            reason = "stable market"
        adjusted = base_price * factor
        minimum, maximum = crop_rules["price_range_rs_per_kg"]
        return _clip(adjusted, float(minimum), float(maximum)), reason

    def _cost_model(self, crop_rules: dict[str, Any], area_ha: float) -> dict[str, float]:
        base_cost = float(crop_rules["base_cost_rs_per_ha"]) * 0.75
        fixed_cost = base_cost * 0.22
        fertilizers = base_cost * 0.18
        pesticides = base_cost * float(crop_rules["chemical_dependency"]) * 0.18
        irrigation = base_cost * water_need_factor(str(crop_rules["water_need"]))
        labour = base_cost * 0.20
        machinery = base_cost * 0.08
        post_harvest = base_cost * 0.09
        subtotal = fixed_cost + fertilizers + pesticides + irrigation + labour + machinery + post_harvest
        buffer = subtotal * 0.12
        total_cost_rs_per_ha = subtotal + buffer
        return {
            "fixed_cost_rs_per_ha": fixed_cost,
            "fertilizer_cost_rs_per_ha": fertilizers,
            "pesticide_cost_rs_per_ha": pesticides,
            "irrigation_cost_rs_per_ha": irrigation,
            "labour_cost_rs_per_ha": labour,
            "machinery_cost_rs_per_ha": machinery,
            "post_harvest_cost_rs_per_ha": post_harvest,
            "buffer_cost_rs_per_ha": buffer,
            "total_cost_rs_per_ha": total_cost_rs_per_ha,
            "total_cost": total_cost_rs_per_ha * area_ha,
        }

    def predict(self, payload: dict[str, Any], top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
        analysis_context = current_analysis_context()
        current_month = int(analysis_context["current_month_number"])
        current_season = str(analysis_context["season"])
        analysis_date = str(analysis_context["analysis_date"])
        prepared = self._fill_defaults(payload)
        class_frame = pd.DataFrame(
            [
                {
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_NUMERIC_FEATURES},
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_CATEGORICAL_FEATURES},
                }
            ]
        )
        X_class = _to_float32_array(self.bundle.classification_preprocessor.transform(class_frame))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lgbm_probs = np.asarray(self.bundle.lgbm_model.predict_proba(X_class))[0]
            stacked_probs = np.asarray(self.bundle.stacking_model.predict_proba(X_class))[0]

        labels = self.bundle.label_encoder.inverse_transform(np.arange(len(stacked_probs)))
        ranked_indices = np.argsort(stacked_probs)[::-1][: self.bundle.metadata["ranking_pool_size"]]
        crop_profiles = self.bundle.metadata["crop_profiles"]
        area_ha = float(prepared["area"])

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for idx in ranked_indices:
            crop_name = str(labels[idx])
            crop_key = crop_name.strip().lower()
            crop_rules = CROP_DATABASE.get(crop_key)
            profile = crop_profiles.get(crop_name, {})
            if not crop_rules or not profile:
                rejected.append({"crop": crop_name, "reason": "missing crop profile or crop rule"})
                continue

            if int(crop_rules["duration_months"]) > 12:
                rejected.append({"crop": crop_name, "reason": "duration exceeds 12 months for a new farmer 1-year analysis"})
                continue

            sowing_months = list(crop_rules["sowing_months"])
            if current_month not in sowing_months:
                rejected.append({"crop": crop_name, "reason": f"outside sowing window for {month_name(current_month)}"})
                continue

            reg_frame = self._build_regression_frame(prepared, crop_name)
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
            climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
            if should_reject:
                rejected.append({"crop": crop_name, "reason": ", ".join(climate_notes)})
                continue

            sowing_penalty = late_sowing_penalty(current_month, sowing_months)
            new_farmer_penalty = float(crop_rules["new_farmer_penalty"])
            total_yield_penalty = min(new_farmer_penalty + climate_penalty + sowing_penalty, 0.58)
            expected_yield_t_ha = max(base_yield_t_ha * (1.0 - total_yield_penalty), 0.15)

            adjusted_price_rs_per_kg, price_reason = self._price_adjustment(crop_name, current_month, crop_rules, prepared)
            harvest_month_number = harvest_month(current_month, int(crop_rules["duration_months"]))
            cost_model = self._cost_model(crop_rules, area_ha)
            revenue_rs_per_ha = expected_yield_t_ha * adjusted_price_rs_per_kg * 1000.0
            profit_rs_per_ha = revenue_rs_per_ha - cost_model["total_cost_rs_per_ha"]
            revenue = revenue_rs_per_ha * area_ha
            profit = profit_rs_per_ha * area_ha

            risk = _clip(
                (float(crop_rules["base_risk"]) * 0.30)
                + (water_need_factor(str(crop_rules["water_need"])) * 0.25)
                + (float(crop_rules["price_volatility"]) * 0.20)
                + (float(crop_rules["new_farmer_penalty"]) * 0.15)
                + (climate_penalty * 0.10),
                0.12,
                0.88,
            )
            sustainability = _clip(
                float(crop_rules["base_sustainability"])
                - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                - (float(crop_rules["chemical_dependency"]) * 0.18)
                - (float(crop_rules["soil_sensitivity"]) * 0.10),
                0.18,
                0.92,
            )

            candidates.append(
                {
                    "crop": crop_name,
                    "classification_probability": float(stacked_probs[idx]),
                    "tabnet_probability": None,
                    "lightgbm_probability": float(lgbm_probs[idx]),
                    "sowing_month": month_name(current_month),
                    "harvest_month": month_name(harvest_month_number),
                    "duration_months": int(crop_rules["duration_months"]),
                    "yield_start_month": month_name(harvest_month_number),
                    "predicted_yield": float(expected_yield_t_ha),
                    "expected_yield_t_ha": float(expected_yield_t_ha),
                    "base_yield_t_ha": float(base_yield_t_ha),
                    "ml_yield_signal_t_ha": float(ml_yield),
                    "adjusted_price_rs_per_kg": float(adjusted_price_rs_per_kg),
                    "market_price_per_ton": float(adjusted_price_rs_per_kg * 1000.0),
                    "total_cost_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "revenue_rs_per_ha": float(revenue_rs_per_ha),
                    "profit_rs_per_ha": float(profit_rs_per_ha),
                    "revenue": float(revenue),
                    "profit": float(profit),
                    "total_cost": float(cost_model["total_cost"]),
                    "risk": float(risk),
                    "risk_score": float(1.0 - risk),
                    "sustainability_score": sustainability,
                    "market_source": "conservative state-season market",
                    "market_type": price_reason,
                    "yield_penalty": float(total_yield_penalty),
                    "late_sowing_penalty": float(sowing_penalty),
                    "new_farmer_penalty": float(new_farmer_penalty),
                    "climate_penalty": float(climate_penalty),
                    "cost_breakdown": cost_model,
                    "advisory_notes": [
                        f"Current month is {month_name(current_month)} and season is {current_season}.",
                        f"Price adjusted for {price_reason}.",
                        "Yield is reduced for new farmer conditions, sowing timing, and climate fit.",
                    ],
                    "rejection_checks_passed": climate_notes or ["seasonal and climate screening passed"],
                }
            )

        if not candidates:
            raise ValueError("No crop is currently suitable for sowing at this date and field condition.")

        profit_scores = _candidate_scores([item["profit_rs_per_ha"] for item in candidates])
        for item, profit_score in zip(candidates, profit_scores):
            item["profit_score"] = float(profit_score)
            item["final_score"] = float((profit_score * 0.5) + ((1.0 - item["risk"]) * 0.3) + (item["sustainability_score"] * 0.2))

        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        top_candidates = candidates[:top_k]
        return {
            "best_crop": top_candidates[0]["crop"],
            "top_crops": top_candidates,
            "training_summary": self.bundle.metadata["training_report"],
            "analysis_context": {
                "analysis_date": analysis_date,
                "current_month": month_name(current_month),
                "season": current_season,
                "area_hectares": area_ha,
                "decision_rule": "Only crops within the current sowing window and under 12 months duration are recommended.",
            },
            "rejected_crops": rejected,
            "used_defaults": {
                "season": current_season,
                "state_name": prepared["state_name"],
                "district_name": prepared["district_name"],
                "crop_year": date.today().year,
                "price_per_ton": prepared["price_per_ton"],
            },
        }
