from __future__ import annotations

from functools import lru_cache

from backend.app.config import MODEL_DIR
from backend.ml.inference import InferenceEngine
from backend.ml.training import train_and_save


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    return InferenceEngine.from_artifacts(MODEL_DIR)


def reload_engine() -> InferenceEngine:
    get_engine.cache_clear()
    return get_engine()


def train_models(data_dir: str | None = None) -> dict:
    report = train_and_save(data_dir=data_dir, model_dir=MODEL_DIR)
    reload_engine()
    return report
