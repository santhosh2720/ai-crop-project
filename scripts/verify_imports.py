from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULES = [
    "backend.ml.config",
    "backend.ml.preprocessing",
    "backend.ml.models",
    "backend.ml.training",
    "backend.ml.inference",
    "backend.app.config",
    "backend.app.schemas",
    "backend.app.services.predictor",
    "backend.app.api.routes",
    "backend.app.main",
]


def main() -> None:
    results = {}
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            results[module_name] = "ok"
        except Exception as exc:
            results[module_name] = f"error: {exc}"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()