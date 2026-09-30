import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
MODEL_DIR = ROOT / "ml" / "model"

CORE_EMOTIONS = ["joy", "anger", "fear", "sadness", "surprise", "disgust", "neutral"]

EMOTION_COLORS = {
    "joy": "#FFD93D",
    "anger": "#FF6B6B",
    "fear": "#9D65C9",
    "sadness": "#6EC1E4",
    "surprise": "#FF9F45",
    "disgust": "#6BCB77",
    "neutral": "#C0C0C0",
}

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

NUMERIC_TO_CORE = {
    0: "joy", 1: "joy", 2: "anger", 3: "anger", 4: "joy", 5: "joy",
    6: "surprise", 7: "surprise", 8: "joy", 9: "sadness", 10: "anger",
    11: "disgust", 12: "sadness", 13: "joy", 14: "fear", 15: "joy",
    16: "sadness", 17: "joy", 18: "joy", 19: "fear", 20: "joy", 21: "joy",
    22: "surprise", 23: "joy", 24: "sadness", 25: "sadness", 26: "surprise",
    27: "neutral",
}

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NAME_RE = re.compile(r"\[NAME\]", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

# Stopword policy shared by training and serving. train.py prints a warning
# if this does not match the configuration of the saved artifacts.
DEFAULT_REMOVE_STOPWORDS = False


def clean_text(text: str, remove_stopwords: bool | None = None) -> str:
    """Lowercase, strip URLs/[NAME], drop punctuation, optionally stopwords.

    ``remove_stopwords=None`` follows ``DEFAULT_REMOVE_STOPWORDS`` so the
    serving path (django predictor.utils) always matches the production
    artifacts without having to know the training configuration.
    """
    if remove_stopwords is None:
        remove_stopwords = DEFAULT_REMOVE_STOPWORDS
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NAME_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if not remove_stopwords:
        return text
    return " ".join(w for w in text.split() if w not in STOPWORDS)


def collapse_labels(label_str: str) -> list[str]:
    """Map a label cell (e.g. "2" or "0,1") to its distinct core emotions."""
    seen = set()
    core = []
    for part in str(label_str).split(","):
        part = part.strip()
        if not part:
            continue
        emotion = NUMERIC_TO_CORE[int(part)]
        if emotion not in seen:
            seen.add(emotion)
            core.append(emotion)
    return core


def is_single_label(label_str: str) -> bool:
    return "," not in str(label_str)


def load_tsv(path, *, remove_stopwords: bool | None = None) -> pd.DataFrame:
    """Read a GoEmotions TSV, drop multi-label rows, collapse and clean."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label", "id"],
        dtype=str,
        keep_default_na=False,
    )
    df = df[df["label"].apply(is_single_label)].copy()
    df["label"] = df["label"].apply(lambda s: NUMERIC_TO_CORE[int(s)])
    df["text"] = df["text"].apply(lambda t: clean_text(t, remove_stopwords=remove_stopwords))
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def get_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """Unfitted word-level TF-IDF vectorizer; fit on train only to avoid leakage."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def class_weights(labels) -> dict:
    """Balanced class weights for the given label list."""
    classes = sorted(set(labels))
    weights = compute_class_weight("balanced", classes=np.asarray(classes), y=labels)
    return dict(zip(classes, weights))


def prepare_data():
    """Load the three splits, fit the word TF-IDF on train, return matrices.

    Artifacts (vectorizer.pkl / model.pkl) are written only by ml/train.py,
    so a standalone run here can never overwrite the production vectorizer
    with one that does not match the trained model.
    """
    train = load_tsv(DATA_DIR / "train.tsv")
    dev = load_tsv(DATA_DIR / "dev.tsv")
    test = load_tsv(DATA_DIR / "test.tsv")

    vectorizer = get_vectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_dev = vectorizer.transform(dev["text"])
    X_test = vectorizer.transform(test["text"])

    return {
        "X_train": X_train, "X_dev": X_dev, "X_test": X_test,
        "y_train": train["label"], "y_dev": dev["label"], "y_test": test["label"],
    }


def main():
    data = prepare_data()
    print(f"Train: {data['X_train'].shape[0]:,} rows, {data['X_train'].shape[1]:,} features")
    print(f"Dev:   {data['X_dev'].shape[0]:,} rows")
    print(f"Test:  {data['X_test'].shape[0]:,} rows")
    print("Note: production artifacts are written by ml/train.py, not here.")


# ---------------------------------------------------------------------------
# Picklable model wrappers. Shared by ml/train.py (saving) and Django's
# predictor.utils (loading); see the module docstring for why they live here.
# ---------------------------------------------------------------------------

class BoostedClassifier:
    """Wraps a fitted classifier to apply dev-tuned decision boosts.

    predict_proba returns boosted, renormalised probabilities so argmax,
    the confidence bars and saved history always stay consistent. Loaded
    by django predictor.utils via model.pkl.
    """

    def __init__(self, estimator, boosts: dict):
        self.estimator = estimator
        self.boosts = boosts

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        probs = self.estimator.predict_proba(X)
        weights = np.array([self.boosts.get(c, 1.0) for c in self.classes_])
        boosted = probs * weights
        return boosted / boosted.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]


class SoftEnsemble:
    """Mean of calibrated member probabilities; duck-types predict_proba."""

    def __init__(self, members):
        self.members = members

    @property
    def classes_(self):
        return self.members[0].classes_

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X):
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]


if __name__ == "__main__":
    main()
