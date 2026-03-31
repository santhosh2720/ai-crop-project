from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.config import (
    DATA_DIR,
    RANDOM_STATE,
    REAL_MARKET_LOOKUP_PATH,
    REAL_PRODUCTION_DATASET_PATH,
    REAL_RECOMMENDATION_DATASET_PATH,
    REPORT_DIR,
)


NUMERIC_FEATURES = [
    "nitrogen",
    "phosphorous",
    "potassium",
    "temperature_c",
    "humidity",
    "ph",
    "rainfall_mm",
    "moisture",
    "area_ha",
]

CATEGORICAL_FEATURES = ["state_name", "district_name", "season"]

PRICE_RANGES_PER_TON = {
    "apple": (45000.0, 70000.0),
    "banana": (18000.0, 30000.0),
    "blackgram": (65000.0, 82000.0),
    "coconut": (45000.0, 70000.0),
    "coffee": (90000.0, 130000.0),
    "grapes": (30000.0, 50000.0),
    "jute": (38000.0, 52000.0),
    "lentil": (55000.0, 75000.0),
    "maize": (20000.0, 28000.0),
    "mango": (28000.0, 45000.0),
    "orange": (25000.0, 42000.0),
    "papaya": (16000.0, 28000.0),
    "rice": (22000.0, 32000.0),
}

YIELD_BASE_PER_HA = {
    "apple": (12.0, 8.0, 18.0),
    "banana": (28.0, 18.0, 40.0),
    "blackgram": (1.0, 0.6, 1.6),
    "coconut": (9.5, 5.0, 14.0),
    "coffee": (1.6, 0.8, 2.5),
    "grapes": (18.0, 10.0, 28.0),
    "jute": (2.3, 1.4, 3.4),
    "lentil": (1.2, 0.7, 1.8),
    "maize": (4.4, 2.5, 6.8),
    "mango": (10.0, 6.0, 15.0),
    "orange": (13.0, 8.0, 20.0),
    "papaya": (30.0, 18.0, 45.0),
    "rice": (3.8, 2.0, 6.2),
}

SEASON_EFFECTS = {
    "Kharif     ": {"temperature_c": 1.2, "humidity": 4.5, "rainfall_mm": 28.0, "price": 0.98, "yield": 1.04},
    "Rabi       ": {"temperature_c": -1.8, "humidity": -2.0, "rainfall_mm": -18.0, "price": 1.02, "yield": 0.98},
    "Summer     ": {"temperature_c": 2.6, "humidity": -6.0, "rainfall_mm": -25.0, "price": 1.03, "yield": 0.95},
    "Whole Year ": {"temperature_c": 0.4, "humidity": 1.5, "rainfall_mm": 8.0, "price": 1.00, "yield": 1.00},
    "Winter     ": {"temperature_c": -2.4, "humidity": 1.0, "rainfall_mm": -10.0, "price": 1.04, "yield": 0.97},
    "Autumn     ": {"temperature_c": 0.2, "humidity": 0.0, "rainfall_mm": 4.0, "price": 1.01, "yield": 1.00},
}


def stable_unit(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stable_uniform(key: str, low: float, high: float) -> float:
    return low + (high - low) * stable_unit(key)


def build_dataset() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    recommendation_df = pd.read_csv(REAL_RECOMMENDATION_DATASET_PATH)
    production_df = pd.read_csv(REAL_PRODUCTION_DATASET_PATH)
    market_df = pd.read_csv(REAL_MARKET_LOOKUP_PATH)

    rec_groups = {
        crop: frame.reset_index(drop=True)
        for crop, frame in recommendation_df.groupby(recommendation_df["crop_norm"].str.lower())
    }

    crop_stats = {}
    global_means = (
        recommendation_df[["nitrogen", "phosphorous", "potassium", "temperature_c", "humidity", "ph", "rainfall_mm"]]
        .mean()
        .to_dict()
    )
    for crop, frame in rec_groups.items():
        crop_stats[crop] = {
            "mean": frame[["nitrogen", "phosphorous", "potassium", "temperature_c", "humidity", "ph", "rainfall_mm"]]
            .mean()
            .to_dict(),
            "std": frame[["nitrogen", "phosphorous", "potassium", "temperature_c", "humidity", "ph", "rainfall_mm"]]
            .std(ddof=0)
            .replace(0, 1.0)
            .to_dict(),
        }

    market_map = {
        str(row["crop_norm"]).lower(): float(row["price_per_ton"])
        for row in market_df.to_dict(orient="records")
        if pd.notna(row.get("price_per_ton"))
    }

    records: list[dict] = []

    for idx, row in production_df.iterrows():
        crop_norm = str(row["crop_norm"]).lower()
        crop_name = str(row["crop"])
        if crop_norm not in rec_groups or crop_norm not in YIELD_BASE_PER_HA:
            continue

        rec_frame = rec_groups[crop_norm]
        sampled = rec_frame.iloc[int(rng.integers(0, len(rec_frame)))]
        stats = crop_stats[crop_norm]

        state_name = str(row["state_name"])
        district_name = str(row["district_name"])
        season = str(row["season"])
        season_effect = SEASON_EFFECTS.get(season, SEASON_EFFECTS["Whole Year "])
        state_key = f"{state_name}|{district_name}"

        climate_anchor = {
            "temperature_c": stable_uniform(f"{state_key}|temp", -2.8, 2.8),
            "humidity": stable_uniform(f"{state_key}|humidity", -7.0, 7.0),
            "rainfall_mm": stable_uniform(f"{state_key}|rain", -30.0, 30.0),
            "nitrogen": stable_uniform(f"{state_key}|n", -9.0, 9.0),
            "phosphorous": stable_uniform(f"{state_key}|p", -8.0, 8.0),
            "potassium": stable_uniform(f"{state_key}|k", -9.0, 9.0),
            "ph": stable_uniform(f"{state_key}|ph", -0.35, 0.35),
        }

        feature_values = {}
        for feature in ["nitrogen", "phosphorous", "potassium", "temperature_c", "humidity", "ph", "rainfall_mm"]:
            base = float(sampled[feature])
            std = float(stats["std"][feature])
            if feature in {"temperature_c", "humidity", "rainfall_mm"}:
                seasonal = float(season_effect[feature])
            else:
                seasonal = 0.0
            noise = rng.normal(0.0, std * 1.45)
            blended = (
                0.52 * base
                + 0.23 * float(stats["mean"][feature])
                + 0.25 * float(global_means[feature])
                + climate_anchor[feature]
                + seasonal
                + noise
            )
            feature_values[feature] = blended

        feature_values["nitrogen"] = float(np.clip(feature_values["nitrogen"], 10, 140))
        feature_values["phosphorous"] = float(np.clip(feature_values["phosphorous"], 5, 145))
        feature_values["potassium"] = float(np.clip(feature_values["potassium"], 5, 210))
        feature_values["temperature_c"] = float(np.clip(feature_values["temperature_c"], 14, 38))
        feature_values["humidity"] = float(np.clip(feature_values["humidity"], 35, 98))
        feature_values["ph"] = float(np.clip(feature_values["ph"], 4.8, 8.1))
        feature_values["rainfall_mm"] = float(np.clip(feature_values["rainfall_mm"], 30, 320))

        moisture = (
            9.0
            + 0.18 * feature_values["humidity"]
            + 0.045 * feature_values["rainfall_mm"]
            - 0.08 * feature_values["temperature_c"]
            + rng.normal(0.0, 3.8)
        )
        moisture = float(np.clip(moisture, 12, 65))

        area_ha = float(max(float(row["area"]), 0.2))

        price_low, price_high = PRICE_RANGES_PER_TON[crop_norm]
        market_base = market_map.get(crop_norm, (price_low + price_high) / 2.0)
        state_multiplier = stable_uniform(f"{state_name}|price", 0.95, 1.05)
        year_multiplier = 1.0 + 0.005 * (int(row["crop_year"]) - 2010) / 10.0
        seasonal_multiplier = float(season_effect["price"])
        stochastic_multiplier = 1.0 + rng.normal(0.0, 0.045)
        price_per_ton = market_base * state_multiplier * seasonal_multiplier * year_multiplier * stochastic_multiplier
        price_per_ton = float(np.clip(price_per_ton, price_low, price_high))
        price_per_ton = round(price_per_ton + idx * 0.01, 2)

        base_yield, min_yield, max_yield = YIELD_BASE_PER_HA[crop_norm]
        nutrient_score = np.mean(
            [
                1.0 - min(abs(feature_values["nitrogen"] - float(stats["mean"]["nitrogen"])) / 60.0, 1.0),
                1.0 - min(abs(feature_values["phosphorous"] - float(stats["mean"]["phosphorous"])) / 60.0, 1.0),
                1.0 - min(abs(feature_values["potassium"] - float(stats["mean"]["potassium"])) / 80.0, 1.0),
            ]
        )
        climate_score = np.mean(
            [
                1.0 - min(abs(feature_values["temperature_c"] - float(stats["mean"]["temperature_c"])) / 10.0, 1.0),
                1.0 - min(abs(feature_values["humidity"] - float(stats["mean"]["humidity"])) / 25.0, 1.0),
                1.0 - min(abs(feature_values["rainfall_mm"] - float(stats["mean"]["rainfall_mm"])) / 120.0, 1.0),
                1.0 - min(abs(feature_values["ph"] - float(stats["mean"]["ph"])) / 1.8, 1.0),
            ]
        )
        suitability = 0.55 * nutrient_score + 0.45 * climate_score
        state_yield_factor = stable_uniform(f"{state_name}|yield", 0.9, 1.08)
        year_yield_factor = 1.0 + 0.003 * (int(row["crop_year"]) - 2010)
        seasonal_yield_factor = float(season_effect["yield"])
        additive_noise = rng.normal(0.0, 0.12 * base_yield)
        target_yield_t_ha = base_yield * (0.72 + 0.42 * suitability) * state_yield_factor * seasonal_yield_factor * year_yield_factor
        target_yield_t_ha = float(np.clip(target_yield_t_ha + additive_noise, min_yield, max_yield))

        records.append(
            {
                "crop": crop_name,
                "state_name": state_name,
                "district_name": district_name,
                "crop_year": int(row["crop_year"]),
                "season": season.strip(),
                "area_ha": round(area_ha, 4),
                "nitrogen": round(feature_values["nitrogen"], 2),
                "phosphorous": round(feature_values["phosphorous"], 2),
                "potassium": round(feature_values["potassium"], 2),
                "temperature_c": round(feature_values["temperature_c"], 2),
                "humidity": round(feature_values["humidity"], 2),
                "ph": round(feature_values["ph"], 2),
                "rainfall_mm": round(feature_values["rainfall_mm"], 2),
                "moisture": round(moisture, 2),
                "price_per_ton": price_per_ton,
                "target_yield_t_ha": round(target_yield_t_ha, 3),
            }
        )

    clean_df = pd.DataFrame.from_records(records)
    return clean_df, {
        "rows": int(len(clean_df)),
        "crops": int(clean_df["crop"].nunique()),
        "states": int(clean_df["state_name"].nunique()),
    }


def validate_dataset(df: pd.DataFrame) -> dict:
    cls_X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    cls_y = df["crop"]
    X_train, X_test, y_train, y_test = train_test_split(
        cls_X,
        cls_y,
        test_size=0.2,
        stratify=cls_y,
        random_state=RANDOM_STATE,
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    cls_model = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=16,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    cls_model.fit(X_train, y_train)
    cls_pred = cls_model.predict(X_test)
    cls_accuracy = float(accuracy_score(y_test, cls_pred))
    cls_f1 = float(f1_score(y_test, cls_pred, average="weighted"))

    reg_X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["crop"]]
    reg_y = df["target_yield_t_ha"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        reg_X,
        reg_y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    reg_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES + ["crop"],
            ),
        ]
    )
    reg_model = Pipeline(
        [
            ("preprocess", reg_pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=260,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    reg_model.fit(X_train_r, y_train_r)
    reg_pred = reg_model.predict(X_test_r)
    reg_rmse = float(np.sqrt(mean_squared_error(y_test_r, reg_pred)))
    reg_r2 = float(r2_score(y_test_r, reg_pred))

    warnings: list[str] = []
    if cls_accuracy > 0.9:
        warnings.append("Classification accuracy is above 0.90; consider adding more overlap/noise if you want a harder benchmark.")
    if cls_accuracy < 0.7:
        warnings.append("Classification accuracy is below 0.70; the synthetic variability may be too strong for a stable training dataset.")

    return {
        "classification_accuracy": cls_accuracy,
        "classification_f1_weighted": cls_f1,
        "regression_rmse": reg_rmse,
        "regression_r2": reg_r2,
        "warnings": warnings,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    clean_df, summary = build_dataset()
    validation = validate_dataset(clean_df)

    clean_path = DATA_DIR / "project_master_clean.csv"
    fixed_path = DATA_DIR / "project_master_fixed.csv"
    clean_df.to_csv(clean_path, index=False)
    clean_df.to_csv(fixed_path, index=False)

    report = {
        "dataset_summary": summary,
        "columns": clean_df.columns.tolist(),
        "validation": validation,
        "leakage_columns_removed": [
            "production",
            "yield",
            "ideal_nitrogen",
            "ideal_phosphorous",
            "ideal_potassium",
            "ideal_temperature_c",
            "ideal_humidity",
            "ideal_ph",
            "ideal_rainfall_mm",
            "estimated_profit_per_hectare",
            "estimated_revenue",
            "recommended_for_live_weather",
        ],
    }
    report_path = REPORT_DIR / "project_master_clean_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"clean_path": str(clean_path), "fixed_path": str(fixed_path), **report}, indent=2))


if __name__ == "__main__":
    main()
