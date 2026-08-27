"""
Epic 5 — Prediction utils boilerplate (E5-T1, T3, T4)
TODO: load model.pkl + vectorizer.pkl, mirror ml/preprocess.clean_text, predict + confidence
"""
from pathlib import Path

# TODO: import pickle, re, string, functools.lru_cache, from django.conf import settings

def clean_text(text: str) -> str:
    """TODO: mirror ml/preprocess.clean_text — must be identical for train/inference"""
    return text  # stub

def load_artifacts():
    """TODO: cached load of MODEL_PATH, VECTORIZER_PATH, LABEL_MAPS_PATH per settings.py"""
    pass

def predict_emotion(text: str) -> dict:
    """
    TODO E5-T4:
    - clean_text(text)
    - vectorizer.transform([cleaned])
    - model.predict / predict_proba or softmax(decision_function)
    - return {predicted: str, confidence: float, scores: {emotion: float}}
    """
    return {"predicted": "neutral", "confidence": 0.0, "scores": {}}  # stub
