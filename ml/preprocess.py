"""
Epic 2 — Preprocessing boilerplate
TODO: implement per CLAUDE.md E2-T1..T2-T6
"""
import re
import string

# Boilerplate constants — fill mapping per CLAUDE.md:196 (27 -> 7)
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

# TODO: populate from GoEmotions numeric IDs (27 = neutral)
NUMERIC_TO_CORE = {
    # Example: 27: "neutral", 2: "anger", 17: "joy"
    # Fill per EDA — merge <500 row classes per CLAUDE.md:242
}

def clean_text(text: str) -> str:
    """TODO E2-T1: lowercase, strip URLs, [NAME], punctuation, stopwords"""
    # TODO: re.sub(r"https?://\S+", "", text.lower())
    # TODO: remove stopwords via nltk
    return text  # stub

def is_single_label(label: str) -> bool:
    """TODO E2-T3: return ',' not in label — drop multilabel rows"""
    return True  # stub

def load_tsv(path: str):
    """TODO: pd.read_csv(path, sep='\t', header=None, names=['text','label','id']) + collapse 27->7"""
    pass

def get_vectorizer():
    """TODO E2-T5: return TfidfVectorizer(max_features=10000, ngram_range=(1,2)) — fit on train only"""
    pass
