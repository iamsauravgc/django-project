# Epic 1 — Data & EDA

> Part of the Emotion Detection Web App. Read `CLAUDE.md` first.

This epic covers **E1-T2 … E1-T6** of the backlog: load the dataset, inspect
it, collapse the 27 GoEmotions emotions into 7 core emotions, and report class
imbalance, text-length distribution, and data-quality issues (nulls,
duplicates, multi-label rows).

E1-T1 (download) was **skipped** — the dataset is already present in
`ml/data/` as `train.tsv`, `dev.tsv`, `test.tsv`.

---

## 1. Dataset

- **Source:** Google GoEmotions (Reddit comments).
- **Format:** tab-separated, 3 columns — `text \t label \t id`.
  - `text` — the comment string.
  - `label` — one or more integer emotion ids (0–27), comma-separated when
    multi-label (e.g. `"0,1"`).
  - `id` — unique per example.
- **Splits:** train 43,410 / dev 5,426 / test 5,427 rows.

All rows were loaded with explicit `string` dtypes so no integer/text parsing
loss occurs.

---

## 2. The 27 → 7 emotion collapse

The GoEmotions labels are numeric ids (0–27, where **27 = neutral**). We map
each id to one of 7 core emotions, defined once in `ml/preprocess.py`
(`NUMERIC_TO_CORE`) so EDA and later training share the same mapping.

| Core emotion | Source GoEmotions ids |
|---|---|
| joy | 0 admiration, 1 amusement, 4 approval, 5 caring, 8 desire, 13 excitement, 15 gratitude, 17 joy, 18 love, 20 optimism, 21 pride, 23 relief |
| anger | 2 anger, 3 annoyance, 10 disapproval |
| fear | 14 fear, 19 nervousness |
| sadness | 9 disappointment, 12 embarrassment, 16 grief, 24 remorse, 25 sadness |
| surprise | 6 confusion, 7 curiosity, 22 realization, 26 surprise |
| disgust | 11 disgust |
| neutral | 27 neutral |

**Decision — ambiguous labels.** The CLAUDE.md reference table did not cover
`admiration(0)`, `approval(4)`, `caring(5)`, and `desire(8)`. Per the
"let EDA decide, merge <500-row classes into the closest emotion" guideline
(CLAUDE.md open question 2), these were mapped to **joy**, the closest
positive/affiliative emotion. This also keeps them out of the already-small
fear/disgust buckets.

**Decision — neutral as 7th class.** Per CLAUDE.md open question 5, `neutral`
is kept as its own class (it is the single most frequent label). This avoids
discarding ~35% of the data and preserves class balance.

**Decision — multi-label rows.** Per CLAUDE.md open question 1, multi-label
rows are **dropped** during training (Epic 2). For EDA reporting, a multi-label
cell like `"0,1"` is de-duplicated to its distinct core emotions
(`collapse_labels()`).

---

## 3. Key findings

All numbers below are from `train.tsv` unless noted.

### 3.1 Class imbalance (collapsed 7-class)

| Core emotion | Train count | Train % |
|---|---:|---:|
| joy | 17,410 | 40.1% |
| neutral | 14,219 | 32.8% |
| anger | 5,579 | 12.9% |
| surprise | 5,367 | 12.4% |
| sadness | 3,263 | 7.5% |
| fear | 726 | 1.7% |
| disgust | 793 | 1.8% |

- **Imbalanced but healthy.** Every core class has **>500 rows** (smallest is
  `fear` at 726), so no further class merging is needed.
- `joy` + `neutral` dominate (~73% combined). `fear` and `disgust` are the
  minority classes — training must use `class_weight="balanced"` (Epic 2,
  E2-T4) and be evaluated with **macro F1**, not accuracy.

### 3.2 Multi-label rows

- **7,102 / 43,410 = 16.36%** of train rows carry more than one label.
- These are dropped before training, leaving **36,308 single-label rows**.
- Dev/test show the same ~16% rate, so the split is consistent.

### 3.3 Nulls, empties, duplicates

| Check | Train | Dev | Test |
|---|---:|---:|---:|
| Null cells | 0 | 0 | 0 |
| Empty text | 0 | 0 | 0 |
| Duplicate ids | 0 | 0 | 0 |
| Duplicate text rows | 183 | 3 | 6 |

- No nulls or empty strings — data is clean at the cell level.
- A small number of **exact-duplicate texts** exist (183 in train). These are
  harmless for EDA; Epic 2 can optionally de-duplicate before training.
- No duplicate `id`s — the multi-label overlap is represented inside single
  rows (comma-separated labels), not as repeated ids.

### 3.4 Text length

Computed on single-label rows (multi-label cells excluded from length stats).

| Metric | Chars | Words |
|---|---:|---:|
| mean | 67.3 | 12.6 |
| median | 64.0 | 12.0 |
| max | 703 | 32 |

- Comments are short (median ~12 words), as expected for Reddit. TF-IDF with
  `max_features=10000` and `ngram_range=(1,2)` (planned for Epic 2) is well
  suited; no truncation needed.

---

## 4. Files

| File | Role |
|---|---|
| `ml/preprocess.py` | Defines `GOEMOTIONS_LABELS`, `NUMERIC_TO_CORE`, `CORE_EMOTIONS`, `EMOTION_COLORS`, and `collapse_labels()` / `is_single_label()`. Text-cleaning and vectorizer stubs remain for Epic 2. |
| `ml/eda.py` | The EDA script: `load_raw`, `inspect`, `label_distribution`, `multilabel_report`, `duplicates_report`, `text_length_stats`, `plot_distributions`, `main`. |
| `ml/eda_artifacts/labels_27.png` | 27-class label distribution bar chart. |
| `ml/eda_artifacts/labels_7.png` | 7-class (collapsed) label distribution bar chart. |
| `ml/eda_artifacts/text_length.png` | Character- and word-length histograms. |

---

## 5. How to run

From the project root:

```bash
python -m ml.eda
```

This prints the full inspection report for all three splits and writes the
three PNG figures into `ml/eda_artifacts/`.

---

## 6. Epic 1 → Epic 2 handoff

- The collapse mapping (`NUMERIC_TO_CORE`) is finalized and reusable.
- `collapse_labels()` returns the core emotions for a label cell; Epic 2 will
  use it in `load_tsv` and **drop** multi-label rows (`is_single_label`).
- Class imbalance confirms the need for `class_weight="balanced"` and macro-F1
  evaluation (E2-T4, E3-T2/T3).
- Next: Epic 2 — text cleaning (`clean_text`), `load_tsv` with collapse +
  multi-label drop, and TF-IDF vectorization fit on train only.
