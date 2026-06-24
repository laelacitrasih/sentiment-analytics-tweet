# Sentiment Analysis of Public Reaction on Platform X

### Entity-Level Sentiment Classification of Public Posts on X (Twitter)

A master's final-semester data-science research project: an **end-to-end, reproducible
sentiment-analysis pipeline** delivered as an interactive **Streamlit** report.

The project classifies public posts on **Platform X (Twitter)** about brands, products
and entities into **positive / negative / neutral** sentiment, and walks through the full
research methodology — from problem definition to interpretation of results.

**Dataset:** [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
(Kaggle) — real, human-annotated tweets, downloaded automatically via `kagglehub`.

---

## Research sections (in the app)

1. Problem Definition
2. Dataset Description
3. Data Sample
4. Problem Formulation
5. Data Collection
6. Data Cleaning
7. Exploratory Data Analysis (EDA)
8. Feature Engineering
9. Model Development
10. Baseline Model
11. Candidate Models
12. Hyperparameter Tuning
13. Cross-Validation
14. Error Analysis
15. Model Comparison
16. Best Model Performance
17. Feature Importance
18. Interpretation of Results
19. Live Demo (interactive prediction)

---

## Project structure

```
.
├── app.py                    # Streamlit research report (run this)
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv           # downloaded & normalized on first run
└── src/
    ├── __init__.py
    ├── load_data.py          # downloads Kaggle dataset, normalizes & samples
    ├── generate_dataset.py   # (legacy) synthetic corpus generator, no longer used
    └── pipeline.py           # cleaning, features, models, tuning, CV, eval
```

---

## Quick start

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) pre-download & build the dataset — the app also does this automatically
python -m src.load_data

# 4. launch the interactive research report
streamlit run app.py
```

> **Note:** the dataset downloads automatically from Kaggle via `kagglehub` on first
> run (no API key needed for this public dataset). It is then cached to
> `data/dataset.csv`.

The app opens at `http://localhost:8501`. Use the **left sidebar** to navigate the
research sections in order.

---

## Methodology highlights

- **Text representation:** TF-IDF (uni-grams + bi-grams, sublinear TF).
- **Baseline:** majority-class `DummyClassifier`.
- **Candidate models:** Logistic Regression, Multinomial Naive Bayes, Linear SVM,
  Random Forest.
- **Model selection:** stratified 5-fold cross-validation (macro-F1).
- **Tuning:** `GridSearchCV` over TF-IDF + Logistic Regression hyperparameters.
- **Interpretation:** per-class coefficient (feature-importance) analysis.
- **Primary metric:** macro-averaged F1 (robust to class imbalance).

---

## Note on the dataset

This study uses the public Kaggle dataset **Twitter Entity Sentiment Analysis** — real
tweets annotated with the sentiment expressed toward a target entity (~74k posts, 32
entities). Preparation (in `src/load_data.py`):

- The **`Irrelevant`** class is dropped (it is a *relevance* label, not a sentiment),
  leaving a clean 3-class problem: positive / negative / neutral.
- Null/empty texts and exact duplicates are removed.
- A **stratified sample of 12,000 posts** is drawn so the interactive app stays
  responsive. Set `SAMPLE_SIZE = None` in `src/load_data.py` to use the full corpus.

Normalized schema: `tweet_id, topic, text, sentiment`.
