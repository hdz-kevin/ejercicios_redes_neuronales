from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


@dataclass
class ModelConfig:
    model_path: str | Path
    class_names: list[str]
    image_size: tuple[int, int]


class FlowerModelService:
    def __init__(self, model_path: str | Path | None = None):
        repo_root = Path(__file__).resolve().parents[1]
        model_file = Path(model_path) if model_path else repo_root / "flower_classifier.keras"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")

        self._config = ModelConfig(
            model_path=model_file,
            class_names=[
                "dandelion",
                "daisy",
                "tulips",
                "sunflowers",
                "roses",
            ],
            image_size=(180, 180),
        )
        self._lock = Lock()
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        with self._lock:
            self._model = tf.keras.models.load_model(str(self._config.model_path))

    @property
    def class_names(self) -> list[str]:
        return self._config.class_names

    @property
    def image_size(self) -> tuple[int, int]:
        return self._config.image_size

    def predict_from_array(self, image_array: np.ndarray) -> dict[str, Any]:
        image = Image.fromarray(image_array.astype(np.uint8))
        image = image.resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = self._model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        all_predictions = [
            {"index": idx, "label": label, "confidence": float(prob)}
            for idx, (label, prob) in enumerate(zip(self.class_names, probabilities))
        ]
        top_k = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)[:3]

        return {
            "top_prediction": {
                "index": predicted_index,
                "label": self.class_names[predicted_index],
                "confidence": confidence,
            },
            "top_k": top_k,
            "label": self.class_names[predicted_index],
            "confidence": confidence,
        }

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        image_array = np.asarray(image)
        return self.predict_from_array(image_array)
