# Emotion Detector

A Django app that classifies any text into one of seven emotions (joy, anger,
fear, sadness, surprise, disgust, neutral) and shows a confidence score for
every class. Each signed-in user gets a history of their own predictions.

The classifier is TF-IDF + LinearSVC trained on Google's GoEmotions dataset
(Reddit comments, 27 labels collapsed down to 7).

## Features

- Register, login, logout (Django built-in auth, all pages protected)
- Text box -> predicted emotion + color-coded confidence bars for all 7 classes
- Predictions saved to the database per user
- History page: your own rows only, with per-emotion summary counts
- Django admin for browsing every prediction
- Seed command that fills the database with realistic demo data

## Tech stack

- Python 3.14, Django 6
- scikit-learn (TfidfVectorizer, LinearSVC), nltk, pandas, numpy
- SQLite (Django default)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

python django_app/manage.py migrate
python django_app/manage.py seed_db      # optional: demo accounts + data
python django_app/manage.py runserver
```

Open http://127.0.0.1:8000/ and register, or use a demo account below.

Note: the first prediction downloads the NLTK English stopword list
(one time, needs network).

## Demo accounts

| Username | Password | Role |
|---|---|---|
| admin | Admin@12345 | superuser, opens /admin/ |
| maya | Demo@12345 | demo user, 13 predictions |
| ravi | Demo@12345 | demo user, 12 predictions |

Re-seed at any time:

```bash
python django_app/manage.py seed_db --reset
```

## Tests

```bash
python django_app/manage.py test accounts predictor
```

28 tests covering auth (register/login/logout/redirects), prediction
(saves rows, scores sum to 1, short input rejected), history isolation
between accounts, and the seed command.

## The model

```bash
python ml/eda.py                    # class balance, text length, chart PNGs
python ml/preprocess.py             # baseline cleaning + TF-IDF, prints shapes
python ml/train.py --dev-only       # compare feature/config variants on dev
python ml/train.py --refit          # tune, save pickles, refit, final test eval
python ml/evaluate.py               # dev report + confusion matrix
```

Pipeline (every choice made on dev; test scored once at the end):

- word (1,3) + char_wb (2,6) TF-IDF, stopwords **kept** (the old filter was
  dropping emotion-critical negations like "not")
- LinearSVC (balanced, weak classes x1.5), C=0.1 selected on dev
- sigmoid calibration (`CalibratedClassifierCV`) for honest probabilities
- per-class decision boosts tuned on dev, applied inside the saved model
  (`ml/preprocess.py`) so verdict and confidence bars stay consistent
- final refit on train+dev

| Stage | Dev macro-F1 |
|---|---:|
| Most-frequent baseline | 0.077 |
| Previous production (word TF-IDF + stopwords) | 0.474 |
| Best feature config (word+char, stopwords kept) | 0.538 |
| + C, class weights, calibration | 0.569 |
| + per-class decision boosts | **0.592** |

**Final test macro-F1: 0.5911** (was 0.5223 before this pass), accuracy 0.653.

Artifacts live in `ml/model/model.pkl` (a `BoostedClassifier` wrapper around
calibrated LinearSVC) and `ml/model/vectorizer.pkl`; `training_scope.txt`
records whether the saved model was fit on train only or train+dev.

LinearSVC has no `predict_proba`, so confidence is the sigmoid-calibrated
probability of the winning class (softmax over `decision_function` remains as
a fallback in `predictor/utils.py` for plain linear models).

## Project structure

```
ml/
  data/            train.tsv, dev.tsv, test.tsv
  model/           model.pkl, vectorizer.pkl
  eda_artifacts/   charts and confusion matrix
  eda.py preprocess.py train.py evaluate.py

django_app/
  manage.py
  config/          settings.py, urls.py, wsgi.py, asgi.py
  accounts/        auth views, templates, tests
  predictor/       prediction views, forms, models, utils, tests
    management/commands/seed_db.py
    templates/     predict.html, history.html
  templates/       base.html
  static/css/      style.css

docs/understanding.md
CLAUDE.md          project brief
```

## Known limitations

- Test macro-F1 is 0.5911, just under the 0.60 goal (missed by 0.009).
  Weak classes: anger (0.48), disgust (0.48), surprise (0.49); disgust and
  fear have fewer than 80 test rows each, so their scores are noisy.
- Single-label output only: one dominant emotion per text.
