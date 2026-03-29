from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.training import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real 13-crop recommendation and yield pipeline.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional data directory containing the real CSV files. Defaults to project data/ then Downloads.",
    )
    args = parser.parse_args()

    report = train_and_save(data_dir=Path(args.data_dir) if args.data_dir else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
