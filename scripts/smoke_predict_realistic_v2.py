from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.inference_realistic_v2 import InferenceEngine


def main() -> None:
    engine = InferenceEngine.from_artifacts()
    payload = {
        "nitrogen": 52,
        "phosphorous": 46,
        "potassium": 41,
        "ph": 6.4,
        "temperature_c": 27.0,
        "humidity": 74.0,
        "rainfall_mm": 118.0,
        "moisture": 33.0,
        "area": 5.5,
        "state_name": "Tamil Nadu",
        "district_name": "Vellore",
        "season": "Kharif",
        "crop_year": 2025,
    }
    print(json.dumps(engine.predict(payload, top_k=3), indent=2))


if __name__ == "__main__":
    main()
