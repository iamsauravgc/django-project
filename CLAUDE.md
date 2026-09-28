# CLAUDE.md — Emotion Detection Web App

## Problem Statement

- People can't objectively understand the emotional tone of their own writing
- Sentiment analysis (positive/negative) is too simple — doesn't capture nuance
- No accessible tool that classifies text into specific emotions (joy, anger, fear, sadness, surprise, disgust) with confidence scores
- Use cases: mental health journaling, customer feedback analysis, content moderation, self-awareness tools

---

## Target User

**Primary:** Tech-savvy individuals aged 18–30 who write — journalers, developers, content creators, students

- **Goals:** Understand the emotional tone of something they wrote or received
- **Frustrations:** Sentiment tools only say "positive/negative" — too coarse
- **Behaviors:** Paste text, want instant result, don't want to sign up for complex tools

---

## Core Features (MVP — Phase 1)

### 1. Authentication
User registration, login, logout using Django's built-in auth. All features gated behind login. Anonymous users redirected to login page.

### 2. Emotion Prediction
User types or pastes any text → model classifies into one of 6 emotions: **Joy / Anger / Fear / Sadness / Surprise / Disgust**. Returns predicted emotion + confidence score for all 6 classes.

### 3. ML Model (Trained Offline)
TF-IDF vectorizer + LinearSVC trained on GoEmotions dataset (58k Reddit comments, 27 emotions collapsed to 6). Saved as `.pkl` files (model + vectorizer). Loaded into Django at runtime.

### 4. Prediction History
Every prediction saved to DB per user — text input, predicted emotion, confidence, timestamp. User can view full history on a dedicated page.

### 5. Results Display
Show: predicted emotion label, confidence % for all 6 emotions, visual indicator (color-coded per emotion).

---

## Nice to Have (Phase 2 — Post MVP)

- **FastAPI layer:** Serve model as REST API; Django consumes `/predict` endpoint
- **LLM explanation:** Gemini API explains why the model predicted that emotion
- **Batch input:** User uploads a CSV of texts → get emotions for all rows
- **Emotion trends:** Dashboard showing user's emotion history over time (chart)
- **Multi-label support:** Text can have multiple emotions simultaneously

---

## Out of Scope (Phase 1)

- No FastAPI — pickle loaded directly in Django view
- No LLM integration
- No real-time analysis
- No mobile app
- No social features
- No multi-label prediction — single dominant emotion only
- No file upload — text input only

---

## Success Metrics

1. Model achieves **F1 score ≥ 0.60** macro average on test set (beats random baseline of 0.17 for 6 classes)
2. User completes full flow (register → login → paste text → see result) in **under 30 seconds**
3. Prediction history correctly isolated per user — no data leaking between accounts

---

## Tech Stack

### Phase 1 (MVP)

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| EDA & visualization | matplotlib, seaborn |
| NLP preprocessing | scikit-learn (TfidfVectorizer), re, nltk |
| ML model | LinearSVC or SGDClassifier |
| Model serialization | pickle |
| Web framework | Django 4.x |
| Auth | Django built-in (`django.contrib.auth`) |
| Database | SQLite (Django default) |
| Dataset | GoEmotions (Google, HuggingFace) |

### Phase 2 (Post MVP)

| Layer | Tool |
|---|---|
| Model API | FastAPI + uvicorn |
| HTTP client | httpx |
| LLM | Google Gemini API |
| Visualization | Chart.js or matplotlib |

---

## Project Structure (Target)

```
emotion_detector/
│
├── ml/
│   ├── data/
│   │   └── go_emotions.csv
│   ├── model/
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│   └── train.py
│
├── django_app/
│   ├── manage.py
│   ├── config/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── predictor/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── forms.py
│   │   └── templates/
│   │       ├── base.html
│   │       ├── predict.html
│   │       ├── result.html
│   │       └── history.html
│   └── accounts/
│       ├── views.py
│       ├── urls.py
│       └── templates/
│           ├── login.html
│           └── register.html
│
└── CLAUDE.md
```

---

## Epic Backlog

### Epic 1 — Data & EDA
- E1-T1: Download GoEmotions dataset from HuggingFace
- E1-T2: Load and inspect — shape, columns, label distribution
- E1-T3: Collapse 27 emotions → 6 core emotions
- E1-T4: Check class imbalance — plot label distribution
- E1-T5: Explore text length distribution
- E1-T6: Check nulls, duplicates, multi-label rows

### Epic 2 — Preprocessing & Feature Engineering
- E2-T1: Clean text — lowercase, remove URLs, punctuation, special chars
- E2-T2: Remove stopwords (nltk)
- E2-T3: Handle multi-label rows — keep dominant emotion or drop
- E2-T4: Handle class imbalance — class_weight='balanced'
- E2-T5: TF-IDF vectorization — fit on train only, transform all splits
- E2-T6: Train/Val/Test split 70/15/15 before vectorization — no leakage

### Epic 3 — Model Training & Evaluation
- E3-T1: Baseline — DummyClassifier, record F1
- E3-T2: Train LinearSVC with class_weight='balanced'
- E3-T3: Evaluate on validation — F1 macro, confusion matrix
- E3-T4: Try SGDClassifier, compare vs LinearSVC
- E3-T5: Tune hyperparameters — GridSearchCV on C parameter
- E3-T6: Final eval on test set — one time only
- E3-T7: Error analysis — which emotions get confused most
- E3-T8: Save `model.pkl` and `vectorizer.pkl`

### Epic 4 — Django Setup & Auth
- E4-T1: Create Django project + apps (`predictor`, `accounts`)
- E4-T2: Configure settings, URLs, templates, static files
- E4-T3: Register, login, logout views + templates
- E4-T4: Protect routes with `@login_required`

### Epic 5 — Prediction Integration
- E5-T1: Load `model.pkl` + `vectorizer.pkl` in Django view
- E5-T2: Build text input form
- E5-T3: Preprocess input text same way as training data
- E5-T4: Predict emotion + confidence scores for all 6 classes
- E5-T5: Result page — emotion label, color-coded confidence bars
- E5-T6: Save prediction to DB per user
- E5-T7: Prediction history page

---

## Emotion Mapping (27 → 6)

| Core Emotion | GoEmotions Labels |
|---|---|
| Joy | joy, amusement, excitement, gratitude, love, optimism, pride, relief |
| Anger | anger, annoyance, disapproval |
| Fear | fear, nervousness |
| Sadness | sadness, grief, remorse, disappointment, embarrassment |
| Surprise | surprise, realization, confusion, curiosity |
| Disgust | disgust |

> Adjust mapping during EDA based on class sizes.

---

## Decision Log

| # | Decision | Alternatives | Reason |
|---|---|---|---|
| 1 | LinearSVC as main model | Naive Bayes, BERT, LogReg | Fast, strong on text, less common than NB in student projects |
| 2 | TF-IDF vectorizer | Bag of Words, Word2Vec, BERT embeddings | Simple, no GPU needed, explainable |
| 3 | 27 → 6 emotion collapse | Keep all 27, binary only | 27 classes too imbalanced; 6 is defensible and interpretable |
| 4 | Django built-in auth | JWT, Firebase, Allauth | Simplest, no extra deps, coursework requirement |
| 5 | GoEmotions dataset | Twitter sentiment, ISEAR | 58k rows, Google-backed, multi-emotion, credible source |


Step	This Project
1. Understand problem	Multi-class text classification, 7 emotions
2. EDA	Label distribution, text length, class imbalance, nulls
3. Clean data	Remove URLs, punctuation, lowercase, drop multi-label rows
4. Feature engineering	TF-IDF vectorization, stopword removal
5. Split data	Already split (train/dev/test) — use as is
6. Scale/normalize	TF-IDF fit on train only, transform dev + test — no leakage
7. Baseline model	DummyClassifier first
8. Train model	LinearSVC on train only
9. Evaluate	F1 macro on dev set (not accuracy — imbalanced classes)
10. Tune hyperparams	GridSearchCV on C parameter, on dev set
11. Final eval	Test set once at the very end
12. Error analysis	Confusion matrix, which emotions get confused most
13. Deploy	Save model.pkl + vectorizer.pkl → Django

---

## Open Questions

1. **Multi-label handling:** Drop multi-label rows or pick dominant emotion by confidence?
Drop multi-label rows. Keeps training clean, no ambiguity. GoEmotions has enough single-label rows (~40k) after dropping.

2. **Emotion collapse:** Use the mapping above or let EDA decide based on class sizes?
Let EDA decide. Use the mapping as a starting point but adjust if any class has under 500 rows — merge it into the closest emotion.

3. **Confidence display:** Raw probabilities or color-coded bar per emotion?
Color-coded bars. Raw probabilities (0.62, 0.21...) mean nothing to a normal user. Colors make it visual and portfolio-worthy.

4. **Deployment:** Local only or deploy to Railway/Render for submission?
Local for MVP submission. Deploy to Railway in Phase 2 — free tier, one command deploy, looks way better on portfolio.

5. **Neutral class:** GoEmotions has a "neutral" label — include as 7th class or drop?
Include as 7th class. It's the most common label in GoEmotions (~35% of data). Dropping it would destroy class balance and throw away a third of your data.





## Warning

No git commit and push until i say and also you need to stop after each epic.


