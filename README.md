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
python ml/eda.py          # class balance, text length, chart PNGs
python ml/preprocess.py   # cleaning + TF-IDF, saves vectorizer.pkl
python ml/train.py        # baseline vs LinearSVC vs SGD, tunes C, saves model.pkl
python ml/evaluate.py     # dev report + confusion matrix
```

| Model | Dev macro-F1 |
|---|---|
| Most-frequent baseline | 0.0768 |
| LinearSVC (balanced) | 0.4776 |
| SGDClassifier (log loss) | 0.5177 |

Final test macro-F1 of the tuned LinearSVC (C=0.1): **0.5223**.

Artifacts live in `ml/model/model.pkl` and `ml/model/vectorizer.pkl`.

LinearSVC has no `predict_proba`, so confidence is a softmax over its
`decision_function`.

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

- Test macro-F1 is 0.5223, below the 0.60 goal. Weak classes: anger,
  surprise, disgust.
- `ml/preprocess.py` lowercases text before stripping the GoEmotions `[NAME]`
  placeholder, so `[NAME]` survives as the word "name". Fixing it means
  retraining.
- Single-label output only: one dominant emotion per text.
