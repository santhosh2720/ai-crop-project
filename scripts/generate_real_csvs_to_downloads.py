from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\santhosh\Downloads")
PRODUCTION_PATH = DOWNLOADS / "crop_production.csv"
RECOMMENDATION_PATH = DOWNLOADS / "Crop_recommendation1.csv"
MARKET_PATH = DOWNLOADS / "Marketwise_Price_Arrival_29-03-2026_07-39-41_AM.csv"

REC_OUTPUT = DOWNLOADS / "real_crop_recommendation_13.csv"
PROD_OUTPUT = DOWNLOADS / "real_crop_production_13.csv"
MARKET_OUTPUT = DOWNLOADS / "real_market_lookup_13.csv"
PROFILE_OUTPUT = DOWNLOADS / "real_crop_reference_profiles_13.csv"
MASTER_OUTPUT = DOWNLOADS / "real_project_master_13.csv"
SUMMARY_OUTPUT = DOWNLOADS / "real_option1_dataset_summary.json"

TARGET_CROPS = [
    "apple",
    "banana",
    "blackgram",
    "coconut",
    "coffee",
    "grapes",
    "jute",
    "lentil",
    "maize",
    "mango",
    "orange",
    "papaya",
    "rice",
]

DISPLAY_NAMES = {
    "apple": "Apple",
    "banana": "Banana",
    "blackgram": "Blackgram",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "grapes": "Grapes",
    "jute": "Jute",
    "lentil": "Lentil",
    "maize": "Maize",
    "mango": "Mango",
    "orange": "Orange",
    "papaya": "Papaya",
    "rice": "Rice",
}

ONLINE_PRICE_REFERENCES = [
    {"crop_norm": "apple", "crop": "Apple", "price_per_ton": 11000.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Apple model project", "source_date": "2026-03-29", "source_url": "https://nhb.gov.in/model-project-reports/Horticulture%20Crops%5Capple%5Capple1.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "banana", "crop": "Banana", "price_per_ton": 6500.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Banana model project", "source_date": "2026-03-29", "source_url": "https://www.nhb.gov.in/report_files/banana/BANANA.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "grapes", "crop": "Grapes", "price_per_ton": 25000.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Grape model project", "source_date": "2026-03-29", "source_url": "https://nhb.gov.in/Horticulture%20Crops%5CGrape%5CGrape1.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "mango", "crop": "Mango", "price_per_ton": 10000.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Mango model project", "source_date": "2026-03-29", "source_url": "https://www.nhb.gov.in/report_files/mango/MANGO.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "orange", "crop": "Orange", "price_per_ton": 12000.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Orange model project", "source_date": "2026-03-29", "source_url": "https://www.nhb.gov.in/report_files/orange/ORANGE.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "papaya", "crop": "Papaya", "price_per_ton": 4500.0, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "reference_project_value", "source_name": "NHB Papaya model project", "source_date": "2026-03-29", "source_url": "https://nhb.gov.in/model-project-reports/Horticulture%20Crops/Papaya/Papaya1.htm", "notes": "Official NHB model-project reference value, not a daily mandi quote."},
    {"crop_norm": "jute", "crop": "Jute", "price_per_ton": 59250.0, "msp_per_ton": 59250.0, "coverage_type": "reference_proxy", "price_type": "msp_reference", "source_name": "PIB Raw Jute MSP 2026-27", "source_date": "2026-02-24", "source_url": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2232109", "notes": "Official MSP for raw jute (TD-3 grade); used as price floor reference."},
    {"crop_norm": "coffee", "crop": "Coffee", "price_per_ton": 325017.2756, "msp_per_ton": np.nan, "coverage_type": "reference_proxy", "price_type": "futures_reference_converted", "source_name": "Coffee Board robusta futures with RBI/FBIL reference rate", "source_date": "2026-03-16", "source_url": "https://coffeeboard.gov.in/", "notes": "Proxy valuation, not a farm-gate quote."},
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    out = out.rename(columns={"temparature": "temperature"})
    return out


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-/&(),.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_crop(value: object) -> str:
    text = normalize_text(value)
    synonyms = {
        "black gram": "blackgram",
        "black gram urd beans whole": "blackgram",
        "urad": "blackgram",
        "paddy common": "rice",
        "copra": "coconut",
    }
    return synonyms.get(text, text)


def build_recommendation() -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = standardize_columns(pd.read_csv(RECOMMENDATION_PATH))
    rec["crop_norm"] = rec["label"].map(normalize_crop)
    rec = rec[rec["crop_norm"].isin(TARGET_CROPS)].copy()
    rec["crop"] = rec["crop_norm"].map(DISPLAY_NAMES)
    rec = rec.rename(columns={"n": "nitrogen", "p": "phosphorous", "k": "potassium", "temperature": "temperature_c", "rainfall": "rainfall_mm"})
    rec["source_dataset"] = "Crop_recommendation1.csv"
    rec = rec[["crop", "crop_norm", "nitrogen", "phosphorous", "potassium", "temperature_c", "humidity", "ph", "rainfall_mm", "label", "source_dataset"]]
    profiles = rec.groupby(["crop", "crop_norm"], as_index=False).agg(
        ideal_nitrogen=("nitrogen", "mean"),
        ideal_phosphorous=("phosphorous", "mean"),
        ideal_potassium=("potassium", "mean"),
        ideal_temperature_c=("temperature_c", "mean"),
        ideal_humidity=("humidity", "mean"),
        ideal_ph=("ph", "mean"),
        ideal_rainfall_mm=("rainfall_mm", "mean"),
        recommendation_rows=("crop", "count"),
    )
    profiles["source_dataset"] = "Crop_recommendation1.csv"
    return rec, profiles


def build_production() -> pd.DataFrame:
    prod = standardize_columns(pd.read_csv(PRODUCTION_PATH))
    prod["crop_norm"] = prod["crop"].map(normalize_crop)
    prod = prod[prod["crop_norm"].isin(TARGET_CROPS)].copy()
    prod["crop"] = prod["crop_norm"].map(DISPLAY_NAMES)
    prod["yield"] = np.where((prod["area"] > 0) & prod["production"].notna(), prod["production"] / prod["area"], np.nan)
    prod["source_dataset"] = "crop_production.csv"
    prod = prod.dropna(subset=["yield", "production"])
    return prod[["crop", "crop_norm", "state_name", "district_name", "crop_year", "season", "area", "production", "yield", "source_dataset"]]


def build_market() -> pd.DataFrame:
    market = pd.read_csv(MARKET_PATH, header=1)
    market.columns = ["commodity_group", "commodity", "msp_qtl", "price_27_qtl", "price_26_qtl", "price_25_qtl", "arrival_27_mt", "arrival_26_mt", "arrival_25_mt"]
    market = market.dropna(subset=["commodity"]).copy()
    market = market[market["commodity"].astype(str).str.lower() != "commodity"].copy()
    for col in ["msp_qtl", "price_27_qtl", "arrival_27_mt"]:
        market[col] = pd.to_numeric(market[col], errors="coerce")
    market["commodity_norm"] = market["commodity"].map(normalize_crop)
    market["crop_norm"] = market["commodity_norm"].replace({"paddy common": "rice", "copra": "coconut", "black gram urd beans whole": "blackgram"})
    market = market[market["crop_norm"].isin(TARGET_CROPS)].copy()
    market["crop"] = market["crop_norm"].map(DISPLAY_NAMES)
    market["price_per_ton"] = market["price_27_qtl"] * 10.0
    market["msp_per_ton"] = market["msp_qtl"] * 10.0
    market["coverage_type"] = "observed_market"
    market["price_type"] = "daily_wholesale_price"
    market["source_name"] = "Local market CSV"
    market["source_date"] = "2026-03-27"
    market["source_url"] = str(MARKET_PATH)
    market["notes"] = "Parsed from user-provided marketwise price report."
    market = market[["crop", "crop_norm", "price_per_ton", "msp_per_ton", "arrival_27_mt", "coverage_type", "price_type", "source_name", "source_date", "source_url", "notes"]]
    combined = pd.concat([market, pd.DataFrame(ONLINE_PRICE_REFERENCES)], ignore_index=True, sort=False)
    priority = {"observed_market": 1, "reference_proxy": 2}
    combined["priority"] = combined["coverage_type"].map(priority).fillna(9)
    combined = combined.sort_values(["crop_norm", "priority"]).drop_duplicates("crop_norm", keep="first")
    return combined.drop(columns=["priority"])


def main() -> None:
    recommendation, profiles = build_recommendation()
    production = build_production()
    market = build_market()
    master = production.merge(profiles, on=["crop", "crop_norm"], how="left").merge(
        market[["crop", "crop_norm", "price_per_ton", "msp_per_ton", "coverage_type", "price_type", "source_name", "source_date"]],
        on=["crop", "crop_norm"],
        how="left",
    )
    master["estimated_profit_per_hectare"] = master["yield"] * master["price_per_ton"]
    master["estimated_revenue"] = master["production"] * master["price_per_ton"]
    master["recommended_for_live_weather"] = True

    recommendation.to_csv(REC_OUTPUT, index=False)
    production.to_csv(PROD_OUTPUT, index=False)
    market.to_csv(MARKET_OUTPUT, index=False)
    profiles.to_csv(PROFILE_OUTPUT, index=False)
    master.to_csv(MASTER_OUTPUT, index=False)

    summary = {
        "recommendation_rows": int(len(recommendation)),
        "production_rows": int(len(production)),
        "market_rows": int(len(market)),
        "master_rows": int(len(master)),
        "files": [str(REC_OUTPUT), str(PROD_OUTPUT), str(MARKET_OUTPUT), str(PROFILE_OUTPUT), str(MASTER_OUTPUT)],
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
