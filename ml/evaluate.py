"""
Epic 3 — Evaluation boilerplate
TODO: classification_report, confusion_matrix, F1 macro on dev (not accuracy)
"""
from pathlib import Path

# TODO: from sklearn.metrics import classification_report, confusion_matrix, f1_score
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
TEST = DATA_DIR / "test.tsv"
DEV = DATA_DIR / "dev.tsv"

def evaluate():
    """TODO E3-T3: load model.pkl + vectorizer.pkl, predict dev, print F1 macro, save confusion.png"""
    pass

if __name__ == "__main__":
    print("[boilerplate] ml/evaluate.py — implement E3-T3..T7")
