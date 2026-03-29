from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "real_crop_recommendation_13.csv"

if not DATASET_PATH.exists():
    DATASET_PATH = Path(r"C:\Users\santhosh\Downloads\real_crop_recommendation_13.csv")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    crop_column = "crop" if "crop" in df.columns else "Crop" if "Crop" in df.columns else None
    summary = {
        "path": str(DATASET_PATH),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "crop_cardinality": int(df[crop_column].nunique()) if crop_column else None,
        "top_crops": df[crop_column].value_counts().head(15).to_dict() if crop_column else {},
        "missing": df.isna().sum().to_dict(),
    }

    print(json.dumps(summary, indent=2))
    print()
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
