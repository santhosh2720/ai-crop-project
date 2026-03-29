from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app


def main() -> None:
    client = TestClient(app)

    metadata = client.get("/api/metadata")
    prediction = client.post(
        "/api/predict",
        json={
            "nitrogen": 90.0,
            "phosphorous": 42.0,
            "potassium": 43.0,
            "ph": 6.5,
            "temperature_c": 21.0,
            "humidity": 82.0,
            "rainfall_mm": 203.0,
            "area": 10.0,
            "price_per_ton": 24819.2,
            "crop_year": 2015,
            "season": "Kharif     ",
            "state_name": "Karnataka",
            "district_name": "TUMKUR",
            "top_k": 3,
        },
    )

    print(
        json.dumps(
            {
                "metadata_status": metadata.status_code,
                "predict_status": prediction.status_code,
                "best_crop": prediction.json().get("best_crop"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
