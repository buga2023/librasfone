from dataclasses import dataclass

import numpy as np

from features import FEATURE_SIZE, sample_features
from landmarks import to_sample


def validate_bundle(bundle: dict) -> dict:
    if not isinstance(bundle, dict) or bundle.get("feature_size") != FEATURE_SIZE:
        raise ValueError("O modelo foi treinado com outra versão das features. Rode python train.py novamente.")
    model, labels = bundle.get("model"), bundle.get("labels")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("O modelo não tem predict_proba. Rode python train.py novamente.")
    if labels is None or len(labels) < 2 or any(not isinstance(label, str) for label in labels):
        raise ValueError("Lista de palavras inválida no modelo. Rode python train.py novamente.")
    # predict_sign usa a posição da maior probabilidade como índice em `labels`,
    # então a ordem precisa ser a mesma das classes do modelo.
    classes = getattr(model, "classes_", None)
    if classes is not None and list(classes) != list(labels):
        raise ValueError("A ordem das palavras difere das classes do modelo. Rode python train.py novamente.")
    return bundle


@dataclass(frozen=True)
class Prediction:

    label: str
    confidence: float
    status: str


def predict_sign(model, labels, frames: np.ndarray, threshold: float) -> Prediction:
    features = sample_features(to_sample(frames))[None, :]
    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    label, confidence = str(labels[best]), float(probabilities[best])
    # Classes que começam com "_" são exemplos negativos (gestos que não são palavras).
    if label.startswith("_"):
        status = "unknown"
    elif confidence < threshold:
        status = "uncertain"
    else:
        status = "accepted"
    return Prediction(label, confidence, status)
