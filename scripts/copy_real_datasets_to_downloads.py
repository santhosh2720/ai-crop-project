from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOWNLOADS = Path(r"C:\Users\santhosh\Downloads")

FILES = [
    DATA_DIR / "real_crop_recommendation_13.csv",
    DATA_DIR / "real_crop_production_13.csv",
    DATA_DIR / "real_market_lookup_13.csv",
    DATA_DIR / "real_crop_reference_profiles_13.csv",
    DATA_DIR / "real_project_master_13.csv",
    DATA_DIR / "real_datasets_13_bundle.zip",
]


def main() -> None:
    for file_path in FILES:
        shutil.copy2(file_path, DOWNLOADS / file_path.name)
    print("copied")


if __name__ == "__main__":
    main()
