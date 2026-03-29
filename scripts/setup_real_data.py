from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOWNLOADS = Path(r"C:\Users\santhosh\Downloads")

FILENAMES = [
    "real_crop_recommendation_13.csv",
    "real_crop_production_13.csv",
    "real_market_lookup_13.csv",
    "real_crop_reference_profiles_13.csv",
    "real_project_master_13.csv",
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        src = DOWNLOADS / filename
        dst = DATA_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
