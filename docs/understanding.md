# Data and EDA

> Part of the Emotion Detection Web App. Read `CLAUDE.md` first.

This document covers loading the dataset, inspecting it, collapsing the 27
GoEmotions emotions into 7 core emotions, and reporting class imbalance,
text-length distribution, and data-quality issues (nulls, duplicates,
multi-label rows).

The download step was **skipped** - the dataset is already present in
`ml/data/` as `train.tsv`, `dev.tsv`, `test.tsv`.

---

## 1. Dataset

- **Source:** Google GoEmotions (Reddit comments).
- **Format:** tab-separated, 3 columns - `text \t label \t id`.
  - `text` - the comment string.
  - `label` - one or more integer emotion ids (0-27), comma-separated when
    multi-label (e.g. `"0,1"`).
  - `id` - unique per example.
- **Splits:** train 43,410 / dev 5,426 / test 5,427 rows.

All rows were loaded with explicit `string` dtypes so no integer/text parsing
loss occurs.

---

## 2. The 27 -> 7 emotion collapse

The GoEmotions labels are numeric ids (0-27, where **27 = neutral**). We map
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

**Decision - ambiguous labels.** The CLAUDE.md reference table did not cover
`admiration(0)`, `approval(4)`, `caring(5)`, and `desire(8)`. Per the
"let EDA decide, merge <500-row classes into the closest emotion" guideline
(CLAUDE.md open question 2), these were mapped to **joy**, the closest
positive/affiliative emotion. This also keeps them out of the already-small
fear/disgust buckets.

**Decision - neutral as 7th class.** Per CLAUDE.md open question 5, `neutral`
is kept as its own class (it is the single most frequent label). This avoids
discarding ~35% of the data and preserves class balance.

**Decision - multi-label rows.** Per CLAUDE.md open question 1, multi-label
rows are **dropped** during training. For EDA reporting, a multi-label
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
  minority classes - training must use `class_weight="balanced"` and be
  evaluated with **macro F1**, not accuracy.

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

- No nulls or empty strings - data is clean at the cell level.
- A small number of **exact-duplicate texts** exist (183 in train). These are
  harmless for EDA; training can optionally de-duplicate beforehand.
- No duplicate `id`s - the multi-label overlap is represented inside single
  rows (comma-separated labels), not as repeated ids.

### 3.4 Text length

Computed on single-label rows (multi-label cells excluded from length stats).

| Metric | Chars | Words |
|---|---:|---:|
| mean | 67.3 | 12.6 |
| median | 64.0 | 12.0 |
| max | 703 | 32 |

- Comments are short (median ~12 words), as expected for Reddit. TF-IDF with
  `max_features=10000` and `ngram_range=(1,2)` (used for training) is well
  suited; no truncation needed.

---

## 4. Files

| File | Role |
|---|---|
| `ml/preprocess.py` | Defines `GOEMOTIONS_LABELS`, `NUMERIC_TO_CORE`, `CORE_EMOTIONS`, `EMOTION_COLORS`, and `collapse_labels()` / `is_single_label()`. Text cleaning and the TF-IDF vectorizer are defined here too. |
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

## 6. Handoff to training

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
Each raw comment is: stripped of URLs (`http(s)://...`, `www....`) and
GoEmotions `[NAME]` placeholders (case-insensitively - the earlier version
lowercased first, so the pattern never matched and `[NAME]` leaked through as
the token "name"); lowercased; reduced to letters, numbers and spaces
(punctuation and other special characters removed); and whitespace-collapsed.
Stopword removal is now **off by default** (`DEFAULT_REMOVE_STOPWORDS = False`):
the dev comparison showed keeping stopwords - negations like "not" are
emotion-bearing - was worth about +0.02 macro-F1. Training and serving both
read the same flag, so they cannot drift. Rows that become empty after
cleaning are discarded.

### Dataset loading (`load_tsv`)
Reads a TSV (`text \t label \t id`), **drops multi-label rows** (a label cell
containing a comma), collapses the remaining single numeric id to its core
emotion via `NUMERIC_TO_CORE`, applies `clean_text`, and returns a DataFrame
with `text` (cleaned) and `label` (core emotion).

### Multi-label handling
Multi-label rows are dropped rather than kept. This was chosen earlier because
GoEmotions still has tens of thousands of single-label rows afterwards, and
dropping avoids ambiguous training targets. After dropping, the training split
has 36,284 usable rows with the current cleaning (down from 43,410 raw).

### Train / validation / test splits
The data ships pre-split by Google (train / dev / test, already disjoint), so
no re-split is performed - the provided splits are used as-is. During model
selection this inherently prevents leakage: the vectorizer is fit only on
train and then applied to dev and test. Only as a **final step** (`--refit`),
after every hyperparameter has already been chosen on dev, is the winning
pipeline re-fit on train+dev; the test set is still never used for any choice.
`ml/model/training_scope.txt` records which scope produced the current
pickles.

### Class imbalance (`class_weights`)
The minority classes (fear, disgust) are far smaller than joy/neutral, so
`class_weights(labels)` returns a `class_weight="balanced"` dictionary
(sklearn `compute_class_weight`) for the classifier to consume at training
time. Evaluation uses macro-F1, not accuracy.

### TF-IDF vectorization (`get_vectorizer`, `prepare_data`)
The word-level vectorizer is a
`TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2,
sublinear_tf=True)` **fit on the training texts only**. `prepare_data()` runs
that baseline pipeline and prints row/feature counts; it deliberately does
**not** write pickles - production artifacts are written only by
`ml/train.py`, whose winning configuration is a `FeatureUnion` of word (1, 3)
and char_wb (2, 6) TF-IDF with a 110,000-feature budget.

Resulting matrix sizes after cleaning:

| Split | Rows | Features (word baseline) |
|---|---:|---:|
| Train | 36,284 | 10,000 |
| Dev | 4,547 | 10,000 |
| Test | 4,588 | 10,000 |

### How to run
```bash
python ml/preprocess.py
```
Prints the per-split row counts (word baseline).

---

## Training & evaluation

Implemented in `ml/train.py` (variant comparison, tuning, artifact writing)
and `ml/evaluate.py` (dev report + confusion matrix). Every design choice -
features, C, class weights, calibration, decision boosts - is made on the
**dev** set; the test set is scored exactly once, at the very end of a final
run. `ml/preprocess.py` also holds the two picklable model wrappers.

### Models compared (dev macro-F1)

| Configuration | Dev macro-F1 |
|---|---:|
| Baseline (most_frequent) | 0.077 |
| word TF-IDF + stopword removal (previous production) | 0.474 |
| word TF-IDF, stopwords kept | 0.496 |
| word (1,2) + char_wb (3,5), stopwords removed | 0.490 |
| word (1,2) + char_wb (3,5), stopwords kept | 0.524 |
| word (1,3) + char_wb (2,6), wide budget, stopwords kept | **0.538** |
| + C=0.1, weak-class weights x1.5, sigmoid calibration | 0.569 |
| + per-class decision boosts | **0.592** |

Reference points: SGDClassifier (log loss) reaches 0.567 raw / 0.592 with
boosts - it ties but never clearly beats the LinearSVC pipeline, so
LinearSVC stays per the project decision. A soft ensemble of calibrated
LinearSVCs at C in {0.05, 0.1, 0.3} also tied the single model (0.591) and
was not adopted (adoption requires winning by more than 0.0005 on dev).

### Hyperparameter tuning
`C` is chosen from {0.03, 0.1, 0.3, 1, 3} by dev macro-F1: **C = 0.1**
(dev 0.561). Class weights are balanced, with the three weakest classes
(anger, surprise, disgust) upweighted **x1.5** (dev sweep: 1.2 was +0.0003,
1.5 was +0.004).

### Calibration and confidence scores
LinearSVC has no `predict_proba`, so the saved model is wrapped in
`CalibratedClassifierCV(method="sigmoid")` (3-fold). Confidence shown in the
UI is the calibrated probability of the winning class; `predictor.utils` still
keeps a softmax-over-`decision_function` fallback for plain linear models.

### Decision boosts
`ml/preprocess.py::BoostedClassifier` multiplies each class's calibrated
probability by a dev-tuned factor before the argmax, then renormalises - so
the verdict, the seven confidence bars and the saved history stay consistent
and still sum to 1. The dev coordinate search settled on: fear 1.3,
surprise 1.4, anger 1.2, disgust 1.0, sadness 1.0, joy 0.8, neutral 0.65.
Boosts were worth **+0.026** macro-F1 on test (0.5651 raw -> 0.5911 served).

### Final test evaluation (served model, scored once)

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| joy | 0.7738 | 0.7511 | 0.7623 |
| anger | 0.5042 | 0.4596 | 0.4809 |
| fear | 0.7027 | 0.6753 | 0.6887 |
| sadness | 0.6902 | 0.4903 | 0.5734 |
| surprise | 0.4911 | 0.4900 | 0.4905 |
| disgust | 0.6591 | 0.3816 | 0.4833 |
| neutral | 0.6217 | 0.7007 | 0.6589 |
| **macro avg** | 0.6347 | 0.5641 | **0.5911** |

Test accuracy is 0.6528. The score improved from 0.5223 to **0.5911**; the
0.60 success target is missed by 0.009. Weak classes remain anger (0.48),
disgust (0.48) and surprise (0.49).

### Error analysis
Dominant confusions on the test set are still with **neutral** - the model
falls back on the majority class for the harder emotions:

| Actual -> Predicted | Count |
|---|---:|
| joy -> neutral | 270 |
| neutral -> joy | 206 |
| anger -> neutral | 175 |
| surprise -> neutral | 145 |
| neutral -> surprise | 125 |
| neutral -> anger | 117 |
| sadness -> neutral | 64 |
| joy -> surprise | 54 |

`python ml/evaluate.py` still writes a dev confusion matrix to
`ml/eda_artifacts/confusion.png`, and prints an in-sample caveat when the
saved artifacts were refit on train+dev.

### How to run
```bash
python ml/train.py --dev-only   # compare variants and tune on dev, no test pass
python ml/train.py --refit      # final: save pickles, refit on train+dev, test eval
python ml/evaluate.py           # dev report + confusion matrix
```
`ml/model/training_scope.txt` records whether the saved artifacts were fit on
train only or train+dev.

### Notes for improvement
The remaining gap to 0.60 (0.009) is structural rather than parametric:
fear (77 test rows) and disgust (76) are tiny classes, and the 27 -> 7
collapse mapping could be revisited for them (the lever the project brief
reserved for EDA). Beyond that, closing the gap means leaving the LinearSVC
decision - an ensemble that actually beats the single model, or a fine-tuned
encoder. The documented feature levers (character n-grams, stopword policy,
calibration) have been exercised.
