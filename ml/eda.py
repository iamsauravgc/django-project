"""
Epic 1 — EDA boilerplate
TODO: implement E1-T2..T6 per CLAUDE.md
"""
from pathlib import Path
# TODO: import pandas, matplotlib, seaborn, from preprocess import CORE_EMOTIONS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
TRAIN = DATA_DIR / "train.tsv"
DEV = DATA_DIR / "dev.tsv"
TEST = DATA_DIR / "test.tsv"

def inspect():
    """TODO:
    - pd.read_csv(TRAIN, sep='\\t', names=['text','label','id'])
    - print shape, label.value_counts(), multilabel %, nulls, dups
    - plot label dist + text length hist to ml/model/
    - collapse 27->7 per CLAUDE.md:196
    """
    pass

if __name__ == "__main__":
    print("[boilerplate] ml/eda.py — run: python ml/eda.py")
    # TODO: inspect()
