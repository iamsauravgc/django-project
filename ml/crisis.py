import re

import numpy as np

CRISIS_LABEL = "sadness"

# How far above the current top score sadness is placed when the
# backoff fires. Kept modest so the bars still look uncertain instead
# of pretending the model was confident.
MARGIN = 0.10

# Matched against normalized text (lowercase, apostrophes removed,
# punctuation stripped): "don't want to live" matches "dont want to
# live". Multi-word phrases are contiguous substrings.
CRISIS_PHRASES = (
    "suicid",                 # covers "suicide", "suicidal"
    "sucide",                 # common typo seen in real user input
    "kill myself",
    "kill my self",
    "killing myself",
    "want to die",
    "end my life",
    "end it all",
    "take my life",
    "not worth living",
    "worthy living",          # covers "life is not worthy living"
    "no reason to live",
    "no point in living",
    "dont want to live",
    "want to be dead",
    "better off dead",
    "better off without me",
    "lost the will to live",
    "give up on life",
)

_APOSTROPHES = ("'", "\u2018", "\u2019", "`")
_STRIP_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """Lowercase, drop apostrophes (don't -> dont), strip punctuation."""
    lowered = text.lower()
    for mark in _APOSTROPHES:
        lowered = lowered.replace(mark, "")
    return _WS_RE.sub(" ", _STRIP_RE.sub(" ", lowered)).strip()


def find_crisis_phrase(text: str):
    """Return the first matching crisis phrase in ``text``, or None."""
    normalized = normalize_for_matching(text)
    for phrase in CRISIS_PHRASES:
        if phrase in normalized:
            return phrase
    return None


def apply_crisis_backoff(probs, labels, text):
    """Force ``CRISIS_LABEL`` above the argmax when ``text`` matches.

    Returns ``(new_probs, matched_phrase)``. No phrase match returns
    the input untouched with ``None``; if sadness already wins, the
    probabilities are returned unchanged.
    """
    phrase = find_crisis_phrase(text)
    if phrase is None:
        return probs, None

    labels = [str(c) for c in labels]
    if CRISIS_LABEL not in labels:
        return probs, phrase

    probabilities = np.asarray(probs, dtype=float).reshape(-1).copy()
    target = labels.index(CRISIS_LABEL)
    top = probabilities.max()
    if probabilities[target] >= top:
        return probabilities, phrase

    probabilities[target] = top + MARGIN
    probabilities /= probabilities.sum()
    return probabilities, phrase
