"""
Shared preprocessing helpers for the emotion detector.

Defines the GoEmotions 27 -> 7 emotion collapse mapping so the EDA script and
the training pipeline use one consistent source of truth. Text cleaning,
vectorization and dataset loading are implemented later in the training stage.
"""
import re
import string

# Core emotions used across the app. "neutral" is kept as a 7th class because
# it is the single most frequent label in GoEmotions.
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

# GoEmotions 28 label names, indexed by their numeric id (0-27).
GOEMOTIONS_LABELS = [
    "admiration",      # 0
    "amusement",       # 1
    "anger",           # 2
    "annoyance",       # 3
    "approval",        # 4
    "caring",          # 5
    "confusion",       # 6
    "curiosity",       # 7
    "desire",          # 8
    "disappointment",  # 9
    "disapproval",     # 10
    "disgust",         # 11
    "embarrassment",   # 12
    "excitement",      # 13
    "fear",            # 14
    "gratitude",       # 15
    "grief",           # 16
    "joy",             # 17
    "love",            # 18
    "nervousness",     # 19
    "optimism",        # 20
    "pride",           # 21
    "realization",     # 22
    "relief",          # 23
    "remorse",         # 24
    "sadness",         # 25
    "surprise",        # 26
    "neutral",         # 27
]

# Numeric GoEmotions id -> core emotion (27 -> 7 collapse).
# Labels not in the original reference table (admiration=0, approval=4,
# caring=5, desire=8) map to "joy" as the closest positive emotion. Every
# core class ends up with more than 500 rows, so no further merging is needed.
NUMERIC_TO_CORE = {
    0: "joy",       # admiration
    1: "joy",       # amusement
    2: "anger",     # anger
    3: "anger",     # annoyance
    4: "joy",       # approval
    5: "joy",       # caring
    6: "surprise",  # confusion
    7: "surprise",  # curiosity
    8: "joy",       # desire
    9: "sadness",   # disappointment
    10: "anger",    # disapproval
    11: "disgust",  # disgust
    12: "sadness",  # embarrassment
    13: "joy",      # excitement
    14: "fear",     # fear
    15: "joy",      # gratitude
    16: "sadness",  # grief
    17: "joy",      # joy
    18: "joy",      # love
    19: "fear",     # nervousness
    20: "joy",      # optimism
    21: "joy",      # pride
    22: "surprise", # realization
    23: "joy",      # relief
    24: "sadness",  # remorse
    25: "sadness",  # sadness
    26: "surprise", # surprise
    27: "neutral",  # neutral
}


def collapse_labels(label_str: str) -> list[str]:
    """Map a GoEmotions label cell (e.g. "2" or "0,1") to core emotion(s).

    Returns the distinct core emotions, de-duplicated. A multi-label cell
    such as "0,1" collapses to ["joy"].
    """
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
    """True when the label cell contains exactly one emotion id."""
    return "," not in str(label_str)


# --- Implemented in the training stage ---

def clean_text(text: str) -> str:
    """TODO: lowercase, strip URLs, [NAME], punctuation and stopwords."""
    return text  # placeholder

def load_tsv(path: str):
    """TODO: read the TSV, collapse 27 labels into 7, drop multi-label rows."""
    pass

def get_vectorizer():
    """TODO: return a TfidfVectorizer(max_features=10000, ngram_range=(1, 2))."""
    pass
