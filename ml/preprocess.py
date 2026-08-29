"""
Shared preprocessing and feature engineering for the emotion detector.

Defines the GoEmotions 27 -> 7 emotion collapse mapping and the text cleaning,
dataset loading and TF-IDF vectorization used before training.
"""
import re
from pathlib import Path

import nltk
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
_NAME_RE = re.compile(r"\[NAME\]")
_SPECIAL_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/[NAME], drop punctuation, remove stopwords."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NAME_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
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


def load_tsv(path) -> pd.DataFrame:
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
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def get_vectorizer() -> TfidfVectorizer:
    """Unfitted TF-IDF vectorizer; fit on train only to avoid leakage."""
    return TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def class_weights(labels) -> dict:
    """Balanced class weights for the given label list."""
    classes = sorted(set(labels))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


def prepare_data():
    """Load splits, fit the vectorizer on train, transform all, save vectorizer.pkl."""
    import pickle

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = load_tsv(DATA_DIR / "train.tsv")
    dev = load_tsv(DATA_DIR / "dev.tsv")
    test = load_tsv(DATA_DIR / "test.tsv")

    vectorizer = get_vectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_dev = vectorizer.transform(dev["text"])
    X_test = vectorizer.transform(test["text"])

    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return {
        "X_train": X_train, "X_dev": X_dev, "X_test": X_test,
        "y_train": train["label"], "y_dev": dev["label"], "y_test": test["label"],
    }


def main():
    data = prepare_data()
    print(f"Train: {data['X_train'].shape[0]:,} rows, {data['X_train'].shape[1]:,} features")
    print(f"Dev:   {data['X_dev'].shape[0]:,} rows")
    print(f"Test:  {data['X_test'].shape[0]:,} rows")
    print(f"Vectorizer saved to {MODEL_DIR / 'vectorizer.pkl'}")


if __name__ == "__main__":
    main()
