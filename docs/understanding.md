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
python ml/eda.py
```

This prints the full inspection report for all three splits and writes the
three PNG figures into `ml/eda_artifacts/`.

---

## 6. Epic 1 → Epic 2 handoff

- The collapse mapping (`NUMERIC_TO_CORE`) is finalized and reusable.
- `collapse_labels()` returns the core emotions for a label cell; the
  preprocessing stage uses it in `load_tsv` and **drops** multi-label rows
  (`is_single_label`).
- Class imbalance confirms the need for `class_weight="balanced"` and macro-F1
  evaluation.

---

## Preprocessing & feature engineering

Implemented in `ml/preprocess.py`. The earlier collapse mapping is reused; the
new work is text cleaning, dataset loading and TF-IDF vectorization.

### Text cleaning (`clean_text`)
Each raw comment is: lowercased; stripped of URLs (`http(s)://…`,
`www.…`) and GoEmotions `[NAME]` placeholders; reduced to letters, numbers and
spaces (punctuation and other special characters removed); whitespace-collapsed;
and filtered to drop English stopwords (NLTK `stopwords`, downloaded on first
use if missing). Rows that become empty after cleaning are discarded.

### Dataset loading (`load_tsv`)
Reads a TSV (`text \t label \t id`), **drops multi-label rows** (a label cell
containing a comma), collapses the remaining single numeric id to its core
emotion via `NUMERIC_TO_CORE`, applies `clean_text`, and returns a DataFrame
with `text` (cleaned) and `label` (core emotion).

### Multi-label handling
Multi-label rows are dropped rather than kept. This was chosen earlier because
GoEmotions still has tens of thousands of single-label rows afterwards, and
dropping avoids ambiguous training targets. After dropping, the training split
has 36,206 usable rows (down from 43,410 raw).

### Train / validation / test splits
The data ships pre-split by Google (train / dev / test, already disjoint), so
no re-split is performed — the provided splits are used as-is. This inherently
prevents leakage: the vectorizer is fit only on train and then applied to dev
and test.

### Class imbalance (`class_weights`)
The minority classes (fear, disgust) are far smaller than joy/neutral, so
`class_weights(labels)` returns a `class_weight="balanced"` dictionary
(sklearn `compute_class_weight`) for the classifier to consume at training
time. Evaluation uses macro-F1, not accuracy.

### TF-IDF vectorization (`get_vectorizer`, `prepare_data`)
A `TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2,
sublinear_tf=True)` is **fit on the training texts only**, then used to
transform dev and test. `prepare_data()` runs the full pipeline and writes the
fitted vectorizer to `ml/model/vectorizer.pkl` for later training and for the
Django runtime.

Resulting matrix sizes after cleaning:

| Split | Rows | Features |
|---|---:|---:|
| Train | 36,206 | 10,000 |
| Dev | 4,535 | 10,000 |
| Test | 4,581 | 10,000 |

### How to run
```bash
python ml/preprocess.py
```
Prints the per-split row counts and saves `ml/model/vectorizer.pkl`.

---

## Training & evaluation

Implemented in `ml/train.py` (training) and `ml/evaluate.py` (dev metrics +
error analysis). The vectorizer from the preprocessing stage is reused, so
there is no leakage: features are fit on train only.

### Models compared
A most-frequent baseline establishes the floor, then LinearSVC (the chosen
model per the project decision) and SGDClassifier are compared on the dev set
with macro-F1.

| Model | Dev macro-F1 |
|---|---:|
| Baseline (most_frequent) | 0.0768 |
| LinearSVC (class_weight="balanced") | 0.4776 |
| SGDClassifier (log_loss, balanced) | 0.5177 |

### Hyperparameter tuning
`GridSearchCV` over `C ∈ [0.1, 1, 10]` (3-fold, `f1_macro`) selected
**C = 0.1** (cv macro-F1 0.4907). The tuned LinearSVC is saved as
`ml/model/model.pkl`. It is kept as the production model even though SGD scored
slightly higher on dev, per the project's model decision.

### Final test evaluation
Scored once, on the held-out test set:

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| joy | 0.8127 | 0.7016 | 0.7531 |
| anger | 0.4645 | 0.3776 | 0.4166 |
| fear | 0.4000 | 0.7792 | 0.5286 |
| sadness | 0.4788 | 0.5676 | 0.5194 |
| surprise | 0.3844 | 0.3415 | 0.3617 |
| disgust | 0.3379 | 0.6447 | 0.4434 |
| neutral | 0.6019 | 0.6681 | 0.6333 |
| **macro avg** | 0.4972 | 0.5829 | **0.5223** |

Test macro-F1 is **0.5223**, above the most-frequent baseline (~0.08) but
**below the 0.60 success target**. Anger, surprise and disgust are the weak
classes.

### Error analysis
`confusion_matrix` on the dev set is saved to `ml/eda_artifacts/confusion.png`.
The dominant confusions are all with **neutral** — the model leans on the
majority class for the harder emotions:

| Actual → Predicted | Count |
|---|---:|
| joy → neutral | 280 |
| surprise → neutral | 192 |
| anger → neutral | 163 |
| neutral → joy | 162 |
| neutral → surprise | 153 |
| neutral → anger | 137 |
| neutral → sadness | 76 |
| joy → surprise | 65 |

### Confidence scores
LinearSVC has no `predict_proba`; per-class confidence is obtained at inference
by applying a softmax to `decision_function` (wired in the prediction stage).

### How to run
```bash
python ml/train.py
python ml/evaluate.py
```
`train.py` prints the baseline/SVC/SGD comparison, tunes C, saves `model.pkl`,
and reports the final test macro-F1. `evaluate.py` prints the dev report and
the most-confused pairs, and writes `confusion.png`.

### Notes for improvement
To reach the 0.60 target: add character n-grams, revisit stopword removal, try
class-probability calibration (`CalibratedClassifierCV`), or enrich the collapse
mapping for the smallest classes (fear, disgust).
