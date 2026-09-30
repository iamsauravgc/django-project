import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from django.conf import settings

from .models import EMOTION_COLORS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.crisis import apply_crisis_backoff  # noqa: E402
from ml.preprocess import clean_text  # noqa: E402  (needs ROOT on sys.path first)


class ArtifactMissing(RuntimeError):
    """Raised when model.pkl / vectorizer.pkl are absent or unreadable."""


@lru_cache(maxsize=1)
def load_artifacts():
    """Load and cache (model, vectorizer) so each process reads them once."""
    model_path = Path(settings.MODEL_PATH)
    vectorizer_path = Path(settings.VECTORIZER_PATH)

    missing = [str(p) for p in (model_path, vectorizer_path) if not p.exists()]
    if missing:
        raise ArtifactMissing(
            "Missing trained artifacts: "
            + ", ".join(missing)
            + ". Run `python ml/train.py` first."
        )

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(vectorizer_path, "rb") as fh:
        vectorizer = pickle.load(fh)
    return model, vectorizer


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (shift by max before exponentiating)."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def predict_emotion(text: str) -> dict:
    """Classify ``text`` into one of the seven core emotions.

    Returns a dict with the cleaned text, the winning label, its confidence,
    every score ranked high-to-low, and the display colour per emotion.
    """
    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(features)).reshape(-1)
        probabilities = _softmax(raw)
    else:  # calibrated models hand back probabilities directly
        probabilities = np.asarray(model.predict_proba(features)).reshape(-1)

    labels = list(model.classes_)

    # Safety net: self-harm phrasing is nearly absent from GoEmotions
    # (see ml/crisis.py), so force sadness above the argmax for those
    # inputs; everything else passes through untouched.
    probabilities, crisis_phrase = apply_crisis_backoff(
        probabilities, labels, text
    )

    order = np.argsort(probabilities)[::-1]

    ranked = {labels[i]: float(probabilities[i]) for i in order}
    predicted = labels[order[0]]
    confidence = ranked[predicted]

    return {
        "cleaned": cleaned,
        "predicted": predicted,
        "predicted_color": EMOTION_COLORS.get(predicted, "#C0C0C0"),
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "scores": ranked,
        "crisis_phrase": crisis_phrase,
        # flat rows so templates never have to do dictionary-key lookups
        "score_list": [
            {
                "emotion": labels[i],
                "score": float(probabilities[i]),
                "color": EMOTION_COLORS.get(labels[i], "#C0C0C0"),
            }
            for i in order
        ],
    }
