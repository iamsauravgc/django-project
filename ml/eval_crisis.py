import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.crisis import apply_crisis_backoff, find_crisis_phrase
from ml.preprocess import DATA_DIR, MODEL_DIR, clean_text

SLICE_PATH = DATA_DIR / "crisis_eval.tsv"


def load_artifacts():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def model_probs(model, vectorizer, text):
    """One row of probabilities via the exact branch serving uses:
    BoostedClassifier has no decision_function, so predictor.utils
    takes the predict_proba path (see django_app/predictor/utils.py)."""
    features = vectorizer.transform([clean_text(text)])
    return model.predict_proba(features).reshape(-1)


def main():
    model, vectorizer = load_artifacts()
    labels = [str(c) for c in model.classes_]
    rows = pd.read_csv(SLICE_PATH, sep="\t", names=["text", "expected", "slice"])

    failures = 0
    stats = {
        "crisis_total": 0, "crisis_model_only": 0, "crisis_served": 0,
        "contrast_total": 0, "contrast_unchanged": 0, "contrast_served": 0,
    }

    print(f"Crisis slice report ({SLICE_PATH.name}, {len(rows)} rows)")
    for row in rows.itertuples(index=False):
        probs = model_probs(model, vectorizer, row.text)
        model_only = labels[probs.argmax()]
        served_probs, phrase = apply_crisis_backoff(probs, labels, row.text)
        served = labels[served_probs.argmax()]
        triggered = phrase is not None

        if row.slice == "crisis":
            stats["crisis_total"] += 1
            stats["crisis_model_only"] += model_only == row.expected
            stats["crisis_served"] += served == row.expected
            ok = triggered and served == row.expected
            if not ok:
                failures += 1
            print(
                f"  [{'ok' if ok else 'FAIL'}] crisis  model-only={model_only:<8}"
                f" served={served:<8} phrase={phrase or '-':<22} {row.text!r}"
            )
        else:
            stats["contrast_total"] += 1
            stats["contrast_unchanged"] += served == model_only
            stats["contrast_served"] += served == row.expected
            ok = not triggered and served == model_only == row.expected
            if not ok:
                failures += 1
            print(
                f"  [{'ok' if ok else 'FAIL'}] contrast served={served:<8}"
                f" model-only={model_only:<8} phrase={phrase or '-':<6} {row.text!r}"
            )

    print()
    print(
        f"crisis  ({stats['crisis_total']}): "
        f"model-only correct {stats['crisis_model_only']}/{stats['crisis_total']}"
        f" -> with backoff {stats['crisis_served']}/{stats['crisis_total']}"
    )
    print(
        f"contrast({stats['contrast_total']}): "
        f"unchanged by backoff {stats['contrast_unchanged']}/{stats['contrast_total']},"
        f" correct {stats['contrast_served']}/{stats['contrast_total']}"
    )
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
