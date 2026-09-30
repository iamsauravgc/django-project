import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ml.preprocess import CORE_EMOTIONS, DATA_DIR, MODEL_DIR, get_vectorizer, load_tsv

ARTIFACT_DIR = ROOT / "ml" / "eda_artifacts"
LABELS = CORE_EMOTIONS
SCOPE_FILE = MODEL_DIR / "training_scope.txt"


def load_artifacts():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def evaluate_dev(model, vectorizer):
    dev = load_tsv(DATA_DIR / "dev.tsv")
    X = vectorizer.transform(dev["text"])
    y_true = dev["label"]
    y_pred = model.predict(X)
    print("Dev set evaluation")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print(f"Dev macro-F1: {f1_score(y_true, y_pred, average='macro', labels=LABELS):.4f}")
    return y_true, y_pred


def error_analysis(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
    )
    ax.set_title("Confusion matrix (dev set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion.png", dpi=120)
    plt.close(fig)

    pairs = [
        (cm[i, j], LABELS[i], LABELS[j])
        for i in range(len(LABELS))
        for j in range(len(LABELS))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    print("\nMost confused pairs (actual -> predicted, count):")
    for count, actual, predicted in pairs[:8]:
        print(f"  {actual} -> {predicted}: {count}")


def main():
    model, vectorizer = load_artifacts()
    scope = SCOPE_FILE.read_text().strip() if SCOPE_FILE.exists() else "train"
    if scope != "train":
        print(
            "NOTE: artifacts were refit on train+dev, so the dev metrics below are\n"
            "      in-sample (optimistic), not held-out. The held-out number is the\n"
            "      test macro-F1 printed at the end of `python ml/train.py --refit`.\n"
        )
    y_true, y_pred = evaluate_dev(model, vectorizer)
    error_analysis(y_true, y_pred)
    print(f"\nConfusion matrix saved to {ARTIFACT_DIR / 'confusion.png'}")


if __name__ == "__main__":
    main()
