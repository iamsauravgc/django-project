"""
Epic 1 + Epic 3 — Training boilerplate
Usage: python ml/train.py --dry-run
"""
import argparse
from pathlib import Path

# TODO: from preprocess import load_tsv, get_vectorizer, CORE_EMOTIONS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
TRAIN = DATA_DIR / "train.tsv"
DEV = DATA_DIR / "dev.tsv"
TEST = DATA_DIR / "test.tsv"
MODEL_DIR = ROOT / "ml" / "model"

def load_splits():
    """TODO E1-T2: load train/dev/test, print shape, label dist, nulls/dups"""
    pass

def train_baseline():
    """TODO E3-T1: DummyClassifier(strategy='stratified') -> F1 macro"""
    pass

def train_svc():
    """TODO E3-T2: LinearSVC(class_weight='balanced') + SGDClassifier compare"""
    pass

def tune_hyperparams():
    """TODO E3-T5: GridSearchCV on C param, cv=3, scoring='f1_macro'"""
    pass

def save_artifacts():
    """TODO E3-T8: pickle.dump(model, open('ml/model/model.pkl','wb')) + vectorizer.pkl"""
    pass

def main(args):
    print("[boilerplate] ml/train.py — implement EDA -> train -> eval per CLAUDE.md Epic 3")
    if args.dry_run:
        print(f"[stub] Would load {TRAIN} ({TRAIN.exists()}) + vectorize train-only (no leakage)")
        return
    # TODO: call load_splits() -> get_vectorizer().fit_transform(train) -> train_baseline() -> train_svc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="stub: load + stats only")
    args = parser.parse_args()
    main(args)
