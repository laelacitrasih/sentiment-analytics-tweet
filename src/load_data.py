"""
Data loader for the research project:
"Sentiment Analysis of Public Reaction on Platform X (Twitter)"

Source: Kaggle — "Twitter Entity Sentiment Analysis"
        (jp797498e/twitter-entity-sentiment-analysis)

The raw file has NO header and 4 columns:
    [tweet_id, topic, sentiment, text]
with four sentiment labels: Positive, Negative, Neutral, Irrelevant.

Design decisions for this study:
  * We DROP the `Irrelevant` class — it is a *relevance* label, not a sentiment, so
    keeping it would pollute the 3-class problem (positive / negative / neutral).
  * We drop null/empty texts and exact-duplicate posts.
  * For an interactive Streamlit app we draw a stratified sample (default 12k posts)
    so training / tuning stay responsive. Set SAMPLE_SIZE = None to use everything.

The normalized result is cached to `data/dataset.csv` with columns:
    [tweet_id, topic, text, sentiment]
"""

from __future__ import annotations

import os

import pandas as pd

KAGGLE_REF = "jp797498e/twitter-entity-sentiment-analysis"
RAW_COLUMNS = ["tweet_id", "topic", "sentiment", "text"]
KEEP_SENTIMENTS = ["positive", "negative", "neutral"]  # 'irrelevant' is dropped
SAMPLE_SIZE = 12000
RANDOM_STATE = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_PATH = os.path.join(ROOT, "data", "dataset.csv")


def _download_raw() -> str:
    """Download the dataset via kagglehub and return the local folder path."""
    import kagglehub

    return kagglehub.dataset_download(KAGGLE_REF)


def _normalize(folder: str, sample_size: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    """Read, clean column names, drop Irrelevant, dedupe and (optionally) sample."""
    train_file = os.path.join(folder, "twitter_training.csv")
    df = pd.read_csv(train_file, header=None, names=RAW_COLUMNS)

    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    df["topic"] = df["topic"].astype(str)

    # drop nulls / empty / the 'irrelevant' class
    df = df[df["text"].str.strip().ne("") & df["text"].str.lower().ne("nan")]
    df = df[df["sentiment"].isin(KEEP_SENTIMENTS)].copy()

    # remove exact-duplicate posts (the corpus has many repeats)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # stratified down-sample for a responsive app
    if sample_size is not None and len(df) > sample_size:
        frac = sample_size / len(df)
        df = (
            df.groupby("sentiment", group_keys=False)
            .sample(frac=frac, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    return df[["tweet_id", "topic", "text", "sentiment"]]


def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return the normalized dataset, building (and caching) it on first use.

    Resolution order:
      1. cached normalized file `data/dataset.csv`
      2. download from Kaggle via kagglehub, normalize, then cache
    """
    if os.path.exists(CANONICAL_PATH) and not force_refresh:
        return pd.read_csv(CANONICAL_PATH)

    os.makedirs(os.path.dirname(CANONICAL_PATH), exist_ok=True)
    folder = _download_raw()
    df = _normalize(folder)
    df.to_csv(CANONICAL_PATH, index=False)
    return df


def main() -> None:
    df = load_dataset(force_refresh=True)
    print(f"Built dataset -> {CANONICAL_PATH}")
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")
    print(f"Topics: {df['topic'].nunique()}")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
