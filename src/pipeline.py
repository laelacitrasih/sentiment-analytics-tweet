"""
NLP / ML pipeline for the sentiment-analysis research project.

This module is UI-agnostic: every function is a plain Python function so it can be
unit-tested or reused outside Streamlit. The Streamlit app wraps the expensive
calls with caching.

Pipeline stages:
    text cleaning  ->  feature engineering  ->  TF-IDF vectorization
    ->  baseline & candidate models  ->  hyperparameter tuning
    ->  cross-validation  ->  evaluation / error analysis / interpretation
"""

from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# 1. Text cleaning
# --------------------------------------------------------------------------- #

# A compact, dependency-free English stop-word list (avoids NLTK download issues).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "their", "his", "her", "so",
    "than", "then", "too", "very", "can", "will", "just", "do", "does", "did", "has",
    "have", "had", "not", "no", "yes", "up", "down", "out", "about", "into", "over",
    "again", "there", "here", "all", "any", "more", "most", "some", "such", "only",
    "own", "same", "s", "t", "us", "am", "im", "u", "ur", "r",
}

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")
ELONGATE_RE = re.compile(r"(.)\1{2,}")  # loveee -> lovee (cap repeats)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]+",
    flags=re.UNICODE,
)


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize a raw social-media post into a clean token string."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)            # keep hashtag word, drop '#'
    t = EMOJI_RE.sub(" ", t)
    t = ELONGATE_RE.sub(r"\1\1", t)         # cap character elongation
    t = NON_ALPHA_RE.sub(" ", t)            # drop numbers/punctuation
    t = MULTISPACE_RE.sub(" ", t).strip()
    if remove_stopwords:
        t = " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 1)
    return t


# --------------------------------------------------------------------------- #
# 2. Data cleaning (dataframe level)
# --------------------------------------------------------------------------- #
def clean_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned dataframe and a report describing what was removed."""
    report = {}
    df = df_raw.copy()
    report["initial_rows"] = len(df)

    # drop empty / whitespace-only text
    blank_mask = df["text"].fillna("").str.strip().eq("")
    report["empty_text_removed"] = int(blank_mask.sum())
    df = df[~blank_mask].copy()

    # drop exact duplicate posts (retweets / copy-paste)
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    report["duplicates_removed"] = before - len(df)

    # cleaned text column
    df["clean_text"] = df["text"].apply(clean_text)

    # drop rows that became empty after cleaning
    empty_after = df["clean_text"].str.strip().eq("")
    report["empty_after_cleaning_removed"] = int(empty_after.sum())
    df = df[~empty_after].copy()

    # normalise missing location (only if the column exists in this dataset)
    if "location" in df.columns:
        df["location"] = df["location"].fillna("").replace("", "Unknown")

    report["final_rows"] = len(df)
    df = df.reset_index(drop=True)
    return df, report


# --------------------------------------------------------------------------- #
# 3. Feature engineering (metadata / structural features)
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out["text"].fillna("")
    out["char_count"] = raw.str.len()
    out["word_count"] = raw.str.split().apply(len)
    out["hashtag_count"] = raw.str.count(r"#\w+")
    out["mention_count"] = raw.str.count(r"@\w+")
    out["has_url"] = raw.str.contains(r"http\S+|www\.", regex=True).astype(int)
    out["exclamation_count"] = raw.str.count("!")
    out["question_count"] = raw.str.count(r"\?")
    out["emoji_count"] = raw.apply(lambda s: len(EMOJI_RE.findall(s)))
    letters = raw.str.count(r"[A-Za-z]").replace(0, np.nan)
    out["uppercase_ratio"] = (raw.str.count(r"[A-Z]") / letters).fillna(0).round(3)
    out["clean_word_count"] = out["clean_text"].str.split().apply(len)
    return out


ENGINEERED_COLS = [
    "char_count", "word_count", "hashtag_count", "mention_count", "has_url",
    "exclamation_count", "question_count", "emoji_count", "uppercase_ratio",
    "clean_word_count",
]


# --------------------------------------------------------------------------- #
# 4. Train / test split
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df["clean_text"].values
    y = df["sentiment"].values
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


# --------------------------------------------------------------------------- #
# 5. Models
# --------------------------------------------------------------------------- #
def make_vectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=5000):
    return TfidfVectorizer(
        ngram_range=ngram_range, min_df=min_df, max_df=max_df,
        max_features=max_features, sublinear_tf=True,
    )


def baseline_pipeline() -> Pipeline:
    """Majority-class baseline: establishes the minimum bar to beat."""
    return Pipeline(
        [
            ("tfidf", make_vectorizer()),
            ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]
    )


def candidate_pipelines() -> dict[str, Pipeline]:
    """The set of candidate models compared in the study."""
    return {
        "Logistic Regression": Pipeline(
            [("tfidf", make_vectorizer()),
             ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))]
        ),
        "Multinomial NB": Pipeline(
            [("tfidf", make_vectorizer()),
             ("clf", MultinomialNB())]
        ),
        "Linear SVM": Pipeline(
            [("tfidf", make_vectorizer()),
             ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            [("tfidf", make_vectorizer(max_features=3000)),
             ("clf", RandomForestClassifier(
                 n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]
        ),
    }


# --------------------------------------------------------------------------- #
# 6. Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_pred": y_pred,
    }


def fit_and_score_all(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit baseline + candidates and return a comparison dataframe."""
    rows = []
    models = {"Baseline (Majority)": baseline_pipeline(), **candidate_pipelines()}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision (macro)": m["precision_macro"],
                "Recall (macro)": m["recall_macro"],
                "F1 (macro)": m["f1_macro"],
                "F1 (weighted)": m["f1_weighted"],
            }
        )
    return pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)


def cross_validate_models(X, y, k: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV (macro-F1) for baseline + candidates."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    models = {"Baseline (Majority)": baseline_pipeline(), **candidate_pipelines()}
    rows = []
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "CV F1 (mean)": scores.mean(),
                "CV F1 (std)": scores.std(),
                "Folds": ", ".join(f"{s:.3f}" for s in scores),
            }
        )
    return pd.DataFrame(rows).sort_values("CV F1 (mean)", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Hyperparameter tuning
# --------------------------------------------------------------------------- #
def tune_logistic_regression(X_train, y_train, k: int = 5):
    """GridSearchCV over TF-IDF + Logistic Regression hyperparameters."""
    pipe = Pipeline(
        [("tfidf", TfidfVectorizer(sublinear_tf=True)),
         ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
    )
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "tfidf__max_df": [0.9, 1.0],
        "clf__C": [0.1, 1.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    return grid


# --------------------------------------------------------------------------- #
# 8. Error analysis & interpretation
# --------------------------------------------------------------------------- #
def confusion_df(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).transpose().round(3)


def top_features_per_class(fitted_pipeline, top_n: int = 15) -> dict[str, pd.DataFrame]:
    """
    Extract the most influential TF-IDF terms per sentiment class from a linear
    model (LogisticRegression / LinearSVC). Returns {class: dataframe}.
    """
    vec = fitted_pipeline.named_steps["tfidf"]
    clf = fitted_pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    feature_names = np.array(vec.get_feature_names_out())
    classes = clf.classes_
    out = {}
    coef = clf.coef_
    # binary case: coef_ has shape (1, n_features)
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    for idx, cls in enumerate(classes):
        weights = coef[idx]
        top_pos = np.argsort(weights)[::-1][:top_n]
        out[cls] = pd.DataFrame(
            {"term": feature_names[top_pos], "weight": weights[top_pos].round(4)}
        )
    return out
