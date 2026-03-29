from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from backend.ml.config import CLASSIFICATION_FEATURES, OPTIONAL_CATEGORICALS, REGRESSION_FEATURES


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col).strip().lower() for col in renamed.columns]
    renamed = renamed.rename(columns={"temparature": "temperature"})
    return renamed.drop_duplicates().reset_index(drop=True)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    numeric_columns = filled.select_dtypes(include=["number"]).columns
    categorical_columns = filled.select_dtypes(exclude=["number"]).columns

    for column in numeric_columns:
        filled[column] = filled[column].fillna(filled[column].median())

    for column in categorical_columns:
        mode = filled[column].mode(dropna=True)
        fallback = mode.iloc[0] if not mode.empty else "unknown"
        filled[column] = filled[column].fillna(fallback)

    return filled


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return fill_missing_values(standardize_columns(df))


def validate_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def build_classification_preprocessor() -> ColumnTransformer:
    numeric_features = [col for col in CLASSIFICATION_FEATURES if col not in OPTIONAL_CATEGORICALS]
    categorical_features = [col for col in OPTIONAL_CATEGORICALS if col in CLASSIFICATION_FEATURES]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def build_regression_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), REGRESSION_FEATURES),
        ]
    )


@dataclass
class PreparedDatasets:
    cleaned_df: pd.DataFrame
    original_missing_yield_mask: pd.Series
    label_encoder: LabelEncoder


def prepare_datasets(raw_df: pd.DataFrame) -> PreparedDatasets:
    standardized = standardize_columns(raw_df)
    validate_columns(
        standardized,
        [
            "crop",
            "season",
            "state",
            "area",
            "fertilizer",
            "pesticide",
            "yield",
            "humidity",
            "moisture",
            "nitrogen",
            "phosphorous",
            "potassium",
            "ph",
            "temperature_c",
            "rainfall_mm",
            "price_per_ton",
            "ideal_temp_c",
            "ideal_rainfall_mm",
        ],
    )
    original_missing_yield_mask = standardized["yield"].isna()
    cleaned_df = fill_missing_values(standardized)
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_df["crop"])
    return PreparedDatasets(
        cleaned_df=cleaned_df,
        original_missing_yield_mask=original_missing_yield_mask,
        label_encoder=label_encoder,
    )


def dataframe_to_float32(array_like: np.ndarray) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float32)