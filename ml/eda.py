"""
Exploratory data analysis for the GoEmotions dataset.

Loads the TSV files in ml/data/ (text, label, id), inspects their structure,
collapses the 27 GoEmotions emotion ids into 7 core emotions, and reports
class imbalance, text-length distribution, nulls, duplicates and multi-label
rows. Figures are saved to ml/eda_artifacts/.

Run from the project root:
    python ml/eda.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless backend so figures are saved to disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.preprocess import (
    CORE_EMOTIONS,
    GOEMOTIONS_LABELS,
    collapse_labels,
    is_single_label,
)

DATA_DIR = ROOT / "ml" / "data"
ARTIFACT_DIR = ROOT / "ml" / "eda_artifacts"

COLUMNS = ["text", "label", "id"]

sns.set_theme(style="whitegrid")


def load_raw(path: Path) -> pd.DataFrame:
    """Load a GoEmotions TSV, keeping text/label/id as strings to avoid parsing loss."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype={"text": "string", "label": "string", "id": "string"},
        keep_default_na=False,
    )


def describe_structure(df: pd.DataFrame) -> dict:
    """Shape, column types, empty cells and duplicate ids/texts for a split."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "empty_text": int((df["text"].str.strip() == "").sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_text": int(df["text"].duplicated().sum()),
    }


def label_counts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns the 27-class distribution and the collapsed 7-class distribution."""
    exploded = df["label"].str.split(",").explode().str.strip().astype(int)
    per_27 = exploded.value_counts().sort_index()
    per_27.index = [GOEMOTIONS_LABELS[i] for i in per_27.index]
    per_27.name = "count"

    collapsed = df["label"].apply(collapse_labels).explode()
    per_7 = collapsed.value_counts().reindex(CORE_EMOTIONS).fillna(0).astype(int)
    per_7.name = "count"
    return per_27, per_7


def report_multilabel(df: pd.DataFrame) -> dict:
    """Share of rows that carry more than one emotion label."""
    multi = ~df["label"].apply(is_single_label)
    return {
        "multi_label_rows": int(multi.sum()),
        "single_label_rows": int((~multi).sum()),
        "multi_label_pct": round(100 * multi.mean(), 2),
    }


def report_duplicates(df: pd.DataFrame) -> dict:
    """Exact-duplicate ids and texts within a split."""
    return {
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_text_rows": int(df["text"].duplicated().sum()),
        "unique_texts": int(df["text"].nunique()),
    }


def text_length_summary(df: pd.DataFrame) -> dict:
    """Character and word length statistics on single-label rows."""
    single = df[df["label"].apply(is_single_label)]
    char_len = single["text"].str.len()
    word_len = single["text"].str.split().apply(len)
    return {
        "char_mean": round(float(char_len.mean()), 1),
        "char_median": float(char_len.median()),
        "char_max": int(char_len.max()),
        "word_mean": round(float(word_len.mean()), 1),
        "word_median": float(word_len.median()),
        "word_max": int(word_len.max()),
    }


def _save_bar(series: pd.Series, title: str, fname: str, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title)
    ax.set_ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = ARTIFACT_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def save_figures(per_27: pd.Series, per_7: pd.Series, df: pd.DataFrame) -> Path:
    """Save label-distribution and text-length figures into ARTIFACT_DIR."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    _save_bar(per_27, "GoEmotions - 27-class label distribution", "labels_27.png",
              sns.color_palette("husl", len(per_27)))
    _save_bar(per_7, "Collapsed - 7-class label distribution", "labels_7.png",
              sns.color_palette("husl", len(per_7)))

    single = df[df["label"].apply(is_single_label)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(single["text"].str.len(), bins=60, color="#6EC1E4")
    ax[0].set_title("Text length (characters)")
    ax[0].set_xlabel("chars")
    ax[0].set_ylabel("count")
    ax[1].hist(single["text"].str.split().apply(len), bins=60, color="#FF9F45")
    ax[1].set_title("Text length (words)")
    ax[1].set_xlabel("words")
    ax[1].set_ylabel("count")
    plt.tight_layout()
    out = ARTIFACT_DIR / "text_length.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "Training set": DATA_DIR / "train.tsv",
        "Development set": DATA_DIR / "dev.tsv",
        "Test set": DATA_DIR / "test.tsv",
    }

    for name, path in splits.items():
        print(f"\n{name}")
        df = load_raw(path)

        structure = describe_structure(df)
        print(f"{structure['rows']:,} rows across {len(structure['columns'])} columns: {structure['columns']}")
        print(f"Column types: {structure['dtypes']}")
        print(f"Null values: {structure['nulls']}  |  empty text fields: {structure['empty_text']}")
        print(f"Duplicate ids: {structure['duplicate_ids']}  |  duplicate texts: {structure['duplicate_text']}")

        per_27, per_7 = label_counts(df)
        print("\n27-class distribution:")
        print(per_27.to_string())
        print("\n7-class (collapsed) distribution:")
        print(per_7.to_string())

        multilabel = report_multilabel(df)
        print(f"\nMulti-label rows: {multilabel['multi_label_rows']:,} "
              f"({multilabel['multi_label_pct']}%)  |  single-label: {multilabel['single_label_rows']:,}")

        duplicates = report_duplicates(df)
        print(f"Duplicates - ids: {duplicates['duplicate_ids']}, texts: {duplicates['duplicate_text_rows']}, "
              f"unique texts: {duplicates['unique_texts']:,}")

        lengths = text_length_summary(df)
        print(f"Text length (chars): mean {lengths['char_mean']}, median {lengths['char_median']}, max {lengths['char_max']}")
        print(f"Text length (words): mean {lengths['word_mean']}, median {lengths['word_median']}, max {lengths['word_max']}")

        if name == "Training set":
            save_figures(per_27, per_7, df)
            print(f"\nFigures saved to {ARTIFACT_DIR}/")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
