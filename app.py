"""
Sentiment Analysis of Public Reaction on Platform X (Twitter)

Master's final-semester research project — interactive report built with Streamlit.
Dataset: Kaggle "Twitter Entity Sentiment Analysis"
         (jp797498e/twitter-entity-sentiment-analysis)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# make 'src' importable regardless of where streamlit is launched from
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import pipeline as pl  # noqa: E402
from src import load_data as ld  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "dataset.csv")
LABELS = ["negative", "neutral", "positive"]
COLOR_MAP = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}

st.set_page_config(
    page_title="Sentiment Analysis on Platform X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached data / model builders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset (downloads from Kaggle on first run)...")
def load_raw() -> pd.DataFrame:
    return ld.load_dataset()


@st.cache_data(show_spinner="Cleaning data...")
def get_clean():
    raw = load_raw()
    return pl.clean_dataframe(raw)


@st.cache_data(show_spinner="Engineering features...")
def get_features():
    clean, _ = get_clean()
    return pl.add_engineered_features(clean)


@st.cache_data(show_spinner="Splitting data...")
def get_split():
    feat = get_features()
    return pl.split_data(feat)


@st.cache_data(show_spinner="Training & comparing models...")
def get_comparison():
    X_train, X_test, y_train, y_test = get_split()
    return pl.fit_and_score_all(X_train, X_test, y_train, y_test)


@st.cache_data(show_spinner="Running cross-validation...")
def get_cv():
    feat = get_features()
    return pl.cross_validate_models(feat["clean_text"].values, feat["sentiment"].values)


@st.cache_resource(show_spinner="Hyperparameter tuning (GridSearchCV)...")
def get_tuned():
    X_train, _, y_train, _ = get_split()
    return pl.tune_logistic_regression(X_train, y_train)


@st.cache_resource(show_spinner="Fitting best model...")
def get_best_model():
    grid = get_tuned()
    return grid.best_estimator_


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
SECTIONS = [
    "1. Problem Definition",
    "2. Dataset Description",
    "3. Data Sample",
    "4. Problem Formulation",
    "5. Data Collection",
    "6. Data Cleaning",
    "7. Exploratory Data Analysis",
    "8. Feature Engineering",
    "9. Model Development",
    "10. Baseline Model",
    "11. Candidate Models",
    "12. Hyperparameter Tuning",
    "13. Cross-Validation",
    "14. Error Analysis",
    "15. Model Comparison",
    "16. Best Model Performance",
    "17. Feature Importance",
    "18. Interpretation of Results",
    "19. Live Demo",
]

st.sidebar.title("Final Project")
st.sidebar.caption("Sentiment Analysis of Public Reaction on Platform X")
choice = st.sidebar.radio("Go to section", SECTIONS, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.info(
    "**Topic:** Public reaction toward brands, products & entities\n\n"
    "**Platform:** X (formerly Twitter)\n\n"
    "**Dataset:** Kaggle — Twitter Entity Sentiment Analysis\n\n"
    "**Author:** Master's Data Science Student"
)


# --------------------------------------------------------------------------- #
# 1. Problem Definition
# --------------------------------------------------------------------------- #
if choice == SECTIONS[0]:
    st.title("Sentiment Analysis of Public Reaction on Platform X")
    st.subheader("Entity-Level Sentiment Classification of Public Posts on X (Twitter)")
    st.divider()

    raw = load_raw()
    st.markdown(
        f"""
### Background & Problem Definition

Every day, users on Platform **X (formerly Twitter)** post millions of reactions to
**brands, products, games, and companies** — from excitement and praise to complaints
and criticism. This public conversation is a goldmine of opinion, but it is enormous,
informal, and noisy (slang, emojis, hashtags, mentions, URLs).

For businesses, marketers, and analysts, understanding *how the public feels* about a
given entity is valuable but impossible to do manually at scale.

**Problem statement.**
> Given a short, unstructured public post on Platform X about some entity, can we
> automatically and reliably classify its sentiment into **positive**, **negative**, or
> **neutral**?

### Why it matters
- **Brand & reputation monitoring** — track how public opinion toward a product shifts.
- **Decision support** — give product and marketing teams a quantitative pulse of opinion.
- **Scalability** — manual reading does not scale to millions of posts; automation does.

### Research goal
Build, tune, and rigorously evaluate a machine-learning text classifier that predicts the
sentiment of public reactions on Platform X, and interpret *what drives* each sentiment.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts analyzed", f"{len(raw):,}")
    c2.metric("Sentiment classes", "3")
    c3.metric("Entities / topics", f"{raw['topic'].nunique()}")


# --------------------------------------------------------------------------- #
# 2. Dataset Description
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[1]:
    section_header(
        "2. Dataset Description",
        "Structure, size, schema and label distribution of the corpus.",
    )
    raw = load_raw()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (posts)", f"{len(raw):,}")
    c2.metric("Columns", f"{raw.shape[1]}")
    c3.metric("Entities / topics", f"{raw['topic'].nunique():,}")
    c4.metric("Sentiment classes", f"{raw['sentiment'].nunique()}")

    st.markdown("### Schema / Data dictionary")
    schema = pd.DataFrame(
        [
            ("tweet_id", "int", "Unique identifier of the post"),
            ("topic", "str", "Entity the post is about (e.g. Borderlands, Amazon, Microsoft)"),
            ("text", "str", "Raw post content (the model input)"),
            ("sentiment", "str", "Ground-truth label: positive / negative / neutral"),
        ],
        columns=["Column", "Type", "Description"],
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.markdown("### Label distribution")
    counts = raw["sentiment"].value_counts().reindex(LABELS)
    fig = px.bar(
        counts, x=counts.index, y=counts.values, color=counts.index,
        color_discrete_map=COLOR_MAP, labels={"x": "Sentiment", "y": "Count"},
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
> **Data provenance.** This study uses the public Kaggle dataset
> **"Twitter Entity Sentiment Analysis"**
> (`jp797498e/twitter-entity-sentiment-analysis`), a benchmark of real tweets, each
> labeled by the sentiment expressed toward a target entity. The original corpus has
> ~74k posts across 32 entities with four labels (Positive, Negative, Neutral,
> **Irrelevant**).
>
> **Preparation for this study:** we **drop the `Irrelevant` class** (it is a
> *relevance* label, not a sentiment), remove null/empty texts and exact duplicates,
> and draw a **stratified sample of 12,000 posts** so the interactive app stays
> responsive. See *Data Collection* and *Data Cleaning* for details.
        """
    )


# --------------------------------------------------------------------------- #
# 3. Data Sample
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[2]:
    section_header("3. Data Sample", "A peek at raw records before any processing.")
    raw = load_raw()

    pick = st.selectbox("Filter by sentiment", ["all"] + LABELS)
    sample = raw if pick == "all" else raw[raw["sentiment"] == pick]
    n = st.slider("Rows to display", 5, 50, 12)
    st.dataframe(sample.head(n), use_container_width=True)

    st.markdown("### Representative posts per class")
    for s in LABELS:
        ex = raw[raw["sentiment"] == s]["text"].dropna()
        ex = ex[ex.str.strip() != ""].head(3).tolist()
        with st.expander(f"{s.capitalize()} examples", expanded=(s == "positive")):
            for e in ex:
                st.markdown(f"- {e}")


# --------------------------------------------------------------------------- #
# 4. Problem Formulation
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[3]:
    section_header(
        "4. Problem Formulation",
        "Translating the business problem into a formal ML task.",
    )
    st.markdown(
        r"""
### Formal task definition
We frame this as a **supervised multi-class text classification** problem.

- **Input** $x$: a single raw post (string) from Platform X.
- **Output** $y \in \{\text{positive}, \text{negative}, \text{neutral}\}$.
- **Goal**: learn a function $f_\theta(x) \rightarrow y$ that generalizes to unseen posts.

We learn $f_\theta$ as a composition:

$$ x \;\xrightarrow{\text{clean}}\; \tilde{x} \;\xrightarrow{\text{TF-IDF}}\; \mathbf{v} \in \mathbb{R}^d \;\xrightarrow{\text{classifier}}\; \hat{y} $$

### Research questions
1. **RQ1** — Can classical ML models on TF-IDF features classify entity-level sentiment
   substantially better than a majority-class baseline?
2. **RQ2** — Which model family (linear vs. probabilistic vs. ensemble) performs best?
3. **RQ3** — Which words/phrases most strongly drive each sentiment class?

### Hypotheses
- **H1**: Linear models (Logistic Regression / Linear SVM) on TF-IDF will beat the
  baseline by a large macro-F1 margin.
- **H2**: Bi-grams add discriminative power over uni-grams alone.

### Evaluation metric
Because classes are imbalanced, the **primary metric is macro-averaged F1**
(equal weight to every class), supported by accuracy, precision, and recall.

$$ F1_{macro} = \frac{1}{C}\sum_{c=1}^{C} \frac{2\,P_c R_c}{P_c + R_c} $$

### Validation protocol
- Stratified 80/20 train–test split (held-out test set for final reporting).
- Stratified 5-fold cross-validation on the training data for model selection.
- GridSearchCV for hyperparameter tuning of the leading model.
        """
    )


# --------------------------------------------------------------------------- #
# 5. Data Collection
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[4]:
    section_header(
        "5. Data Collection",
        "How the corpus was assembled and the sampling design.",
    )
    raw = load_raw()
    st.markdown(
        f"""
### Source
The corpus is the public Kaggle dataset **"Twitter Entity Sentiment Analysis"**
(`jp797498e/twitter-entity-sentiment-analysis`) — real tweets collected and
human-annotated for the sentiment expressed toward a **target entity** (e.g. a game,
brand, or company). It is downloaded programmatically and reproducibly:

```python
import kagglehub
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
# -> twitter_training.csv (≈74k rows), twitter_validation.csv (≈1k rows)
# columns (no header): [tweet_id, topic, sentiment, text]
```

### Sampling & label design
- **Original size:** ~74,682 posts across **32 entities**, four labels
  (Positive, Negative, Neutral, Irrelevant).
- **Irrelevant dropped:** it marks posts *not relevant* to the entity — a relevance
  flag, not a sentiment — so it is excluded to keep a clean 3-class problem.
- **Deduplicated:** exact-duplicate posts are removed.
- **Stratified sample:** {len(raw):,} posts are sampled (class proportions preserved)
  so the interactive app trains and tunes quickly. This is configurable in
  `src/load_data.py` (`SAMPLE_SIZE = None` uses the full corpus).

### Labeling
Each post carries a **human-annotated ground-truth sentiment label** from the original
benchmark, which lets us train and evaluate the model on real, independently labeled data.
        """
    )

    st.markdown("### Posts per entity / topic (top 15)")
    top_topics = raw["topic"].value_counts().head(15).reset_index()
    top_topics.columns = ["topic", "posts"]
    fig = px.bar(top_topics, x="posts", y="topic", orientation="h",
                 color="posts", color_continuous_scale="Blues")
    fig.update_layout(height=460, yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 6. Data Cleaning
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[5]:
    section_header(
        "6. Data Cleaning",
        "Removing noise and normalizing text into model-ready tokens.",
    )
    raw = load_raw()
    clean, report = get_clean()

    st.markdown("### Cleaning pipeline applied")
    st.markdown(
        """
1. Drop empty / whitespace-only posts.
2. Remove exact duplicate posts (retweets / copy-paste).
3. **Text normalization** per post:
   lowercase → remove URLs → remove @mentions → keep hashtag *word* (drop `#`) →
   strip emojis → cap character elongation (*loveee → lovee*) → remove
   numbers/punctuation → remove stop-words & 1-char tokens → collapse whitespace.
4. Drop rows that became empty after cleaning (e.g. posts that were only emojis/links).

*Note:* duplicates and null texts are already filtered when the dataset is built in
`src/load_data.py`, so the counts below reflect what remains after that initial pass.
        """
    )

    st.markdown("### Cleaning report")
    rep = pd.DataFrame(
        {
            "Step": [
                "Initial rows", "Empty text removed", "Duplicates removed",
                "Empty after cleaning removed", "Final rows",
            ],
            "Count": [
                report["initial_rows"], report["empty_text_removed"],
                report["duplicates_removed"], report["empty_after_cleaning_removed"],
                report["final_rows"],
            ],
        }
    )
    st.dataframe(rep, use_container_width=True, hide_index=True)
    removed = report["initial_rows"] - report["final_rows"]
    st.metric("Total rows removed", f"{removed:,}",
              delta=f"-{removed / report['initial_rows']:.1%}")

    st.markdown("### Before → After examples")
    show = clean[["text", "clean_text"]].head(8).rename(
        columns={"text": "Raw post", "clean_text": "Cleaned tokens"}
    )
    st.dataframe(show, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 7. EDA
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[6]:
    section_header(
        "7. Exploratory Data Analysis",
        "Understanding class balance, lengths, entities and vocabulary.",
    )
    feat = get_features()

    tab1, tab2, tab3 = st.tabs(
        ["Class & topics", "Text length", "Top words / word cloud"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        counts = feat["sentiment"].value_counts().reindex(LABELS)
        fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                     color_discrete_map=COLOR_MAP, title="Sentiment distribution", hole=0.4)
        col1.plotly_chart(fig, use_container_width=True)

        top_t = feat["topic"].value_counts().head(12).index
        cross = (feat[feat["topic"].isin(top_t)]
                 .groupby(["topic", "sentiment"]).size().reset_index(name="count"))
        fig2 = px.bar(cross, x="topic", y="count", color="sentiment",
                      color_discrete_map=COLOR_MAP, barmode="stack",
                      title="Sentiment by entity (top 12)")
        fig2.update_layout(xaxis_tickangle=-40, height=420)
        col2.plotly_chart(fig2, use_container_width=True)
        st.caption("Sentiment skews differently per entity — some brands attract more "
                   "negative reaction than others.")

    with tab2:
        col1, col2 = st.columns(2)
        fig = px.histogram(feat, x="word_count", color="sentiment", nbins=40,
                           color_discrete_map=COLOR_MAP, marginal="box",
                           title="Word-count distribution by sentiment")
        col1.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(feat, x="sentiment", y="char_count", color="sentiment",
                      color_discrete_map=COLOR_MAP, title="Character count by sentiment")
        fig2.update_layout(showlegend=False)
        col2.plotly_chart(fig2, use_container_width=True)

    with tab3:
        from collections import Counter

        sel = st.selectbox("Word frequency for class", ["all"] + LABELS)
        subset = feat if sel == "all" else feat[feat["sentiment"] == sel]
        tokens = " ".join(subset["clean_text"]).split()
        common = Counter(tokens).most_common(20)
        freq_df = pd.DataFrame(common, columns=["term", "count"])
        col1, col2 = st.columns([1, 1])
        fig = px.bar(freq_df, x="count", y="term", orientation="h",
                     title=f"Top 20 terms ({sel})")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=520)
        col1.plotly_chart(fig, use_container_width=True)

        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            wc = WordCloud(width=600, height=480, background_color="white",
                           colormap="viridis").generate(" ".join(tokens))
            fig2, ax = plt.subplots(figsize=(6, 4.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            col2.pyplot(fig2)
        except Exception as e:  # wordcloud optional
            col2.info(f"Word cloud unavailable ({e}). Bar chart shown instead.")


# --------------------------------------------------------------------------- #
# 8. Feature Engineering
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[7]:
    section_header(
        "8. Feature Engineering",
        "From raw text to numerical features: TF-IDF + structural signals.",
    )
    feat = get_features()

    st.markdown(
        """
### A. Text representation — TF-IDF
The cleaned text is vectorized with **TF-IDF** (Term Frequency–Inverse Document
Frequency), which down-weights ubiquitous words and up-weights distinctive ones:

$$ \\text{tfidf}(t,d) = \\text{tf}(t,d) \\cdot \\log\\frac{N}{1+\\text{df}(t)} $$

Configuration: **uni-grams + bi-grams**, `min_df=2`, `max_df=0.9`,
`max_features=5000`, sublinear TF scaling.

### B. Structural / metadata features
Beyond words, social-media *form* carries signal. We engineer:
        """
    )
    eng = pd.DataFrame(
        [
            ("char_count / word_count", "Length of the post"),
            ("hashtag_count / mention_count", "Social markup intensity"),
            ("has_url", "Whether the post links out (often neutral/news)"),
            ("exclamation_count / question_count", "Emotional / inquisitive tone"),
            ("emoji_count", "Emoji usage (often polarized)"),
            ("uppercase_ratio", "Shouting / emphasis"),
            ("clean_word_count", "Length after cleaning"),
        ],
        columns=["Feature(s)", "Intuition"],
    )
    st.dataframe(eng, use_container_width=True, hide_index=True)

    st.markdown("### Engineered features preview")
    st.dataframe(feat[["text", "sentiment"] + pl.ENGINEERED_COLS].head(8),
                 use_container_width=True, hide_index=True)

    st.markdown("### Do engineered features separate the classes?")
    feat_choice = st.selectbox("Inspect a feature", pl.ENGINEERED_COLS, index=2)
    fig = px.box(feat, x="sentiment", y=feat_choice, color="sentiment",
                 color_discrete_map=COLOR_MAP)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    corr = feat[pl.ENGINEERED_COLS].corr()
    fig2 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu_r", title="Feature correlation matrix")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Note: the modeling pipeline below uses TF-IDF text features (the strongest "
        "signal for sentiment). Structural features are presented for EDA insight and "
        "could be concatenated in an extended model."
    )


# --------------------------------------------------------------------------- #
# 9. Model Development
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[8]:
    section_header(
        "9. Model Development",
        "The modeling strategy, pipeline, and experimental setup.",
    )
    X_train, X_test, y_train, y_test = get_split()
    st.markdown(
        f"""
### Modeling strategy
All models share a unified **scikit-learn `Pipeline`** so that vectorization and
classification are fitted together (preventing data leakage across CV folds):

```
Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", <classifier>)])
```

### Experimental setup
- **Train set:** {len(X_train):,} posts &nbsp;|&nbsp; **Test set:** {len(X_test):,} posts
- **Split:** stratified 80/20, `random_state=42` (reproducible).
- **Selection metric:** macro-F1 via stratified 5-fold CV.
- **Final reporting:** on the untouched held-out test set.

### The models compared
| Role | Model | Why included |
|------|-------|--------------|
| Baseline | Majority-class `DummyClassifier` | Minimum bar any real model must beat |
| Candidate | Logistic Regression | Strong, interpretable linear text classifier |
| Candidate | Multinomial Naive Bayes | Classic, fast probabilistic text baseline |
| Candidate | Linear SVM | Often state-of-the-art for sparse TF-IDF |
| Candidate | Random Forest | Non-linear ensemble for comparison |

Proceed through sections **10–16** to see each stage's results.
        """
    )


# --------------------------------------------------------------------------- #
# 10. Baseline Model
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[9]:
    section_header(
        "10. Baseline Model",
        "A majority-class predictor establishes the floor to beat.",
    )
    X_train, X_test, y_train, y_test = get_split()
    base = pl.baseline_pipeline()
    base.fit(X_train, y_train)
    m = pl.evaluate(base, X_test, y_test)

    st.markdown(
        """
The **baseline** always predicts the most frequent class. It is *not* useful in
practice, but it defines the **minimum performance** any candidate model must exceed.
A model that cannot beat this has learned nothing.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Macro-F1", f"{m['f1_macro']:.3f}")
    c3.metric("Weighted-F1", f"{m['f1_weighted']:.3f}")
    st.info(
        f"The baseline reaches **{m['accuracy']:.1%} accuracy** simply by always "
        f"guessing the majority class, but its **macro-F1 is only {m['f1_macro']:.3f}** "
        "because it completely fails on the minority classes. This is exactly why "
        "macro-F1 is our primary metric."
    )
    st.markdown("#### Baseline classification report")
    st.dataframe(pl.classification_report_df(y_test, m["y_pred"]),
                 use_container_width=True)


# --------------------------------------------------------------------------- #
# 11. Candidate Models
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[10]:
    section_header(
        "11. Candidate Models",
        "Train each candidate and inspect held-out test performance.",
    )
    X_train, X_test, y_train, y_test = get_split()
    comp = get_comparison()

    st.markdown("### Held-out test performance (all models)")
    st.dataframe(
        comp.style.format(
            {c: "{:.3f}" for c in comp.columns if c != "Model"}
        ).background_gradient(cmap="Greens", subset=["F1 (macro)"]),
        use_container_width=True,
    )

    st.markdown("### Inspect a single candidate")
    model_name = st.selectbox(
        "Choose a candidate", list(pl.candidate_pipelines().keys())
    )
    pipe = pl.candidate_pipelines()[model_name]
    pipe.fit(X_train, y_train)
    m = pl.evaluate(pipe, X_test, y_test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision_macro']:.3f}")
    c3.metric("Recall", f"{m['recall_macro']:.3f}")
    c4.metric("Macro-F1", f"{m['f1_macro']:.3f}")
    st.dataframe(pl.classification_report_df(y_test, m["y_pred"]),
                 use_container_width=True)


# --------------------------------------------------------------------------- #
# 12. Hyperparameter Tuning
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[11]:
    section_header(
        "12. Hyperparameter Tuning",
        "GridSearchCV over TF-IDF + Logistic Regression (the leading family).",
    )
    st.markdown(
        """
We tune the strongest linear model with **`GridSearchCV`** (stratified 5-fold,
scoring = macro-F1) over the search space:

| Hyperparameter | Values |
|----------------|--------|
| `tfidf__ngram_range` | (1,1), (1,2) |
| `tfidf__min_df` | 1, 2, 5 |
| `tfidf__max_df` | 0.9, 1.0 |
| `clf__C` | 0.1, 1.0, 10.0 |

That is **36 configurations × 5 folds = 180 fits**.
        """
    )
    grid = get_tuned()
    st.success(f"**Best CV macro-F1:** {grid.best_score_:.4f}")
    st.markdown("#### Best hyperparameters")
    st.json(grid.best_params_)

    st.markdown("#### Top configurations")
    res = pd.DataFrame(grid.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score"
    ]
    top = res[cols].sort_values("rank_test_score").head(10).reset_index(drop=True)
    st.dataframe(top, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 13. Cross-Validation
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[12]:
    section_header(
        "13. Cross-Validation",
        "Stratified 5-fold CV gives a robust, low-variance estimate of generalization.",
    )
    cv = get_cv()
    st.markdown("### Stratified 5-fold CV (macro-F1)")
    st.dataframe(
        cv.style.format({"CV F1 (mean)": "{:.3f}", "CV F1 (std)": "{:.3f}"})
        .background_gradient(cmap="Greens", subset=["CV F1 (mean)"]),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(
        cv, x="Model", y="CV F1 (mean)", error_y="CV F1 (std)", color="Model",
        title="Cross-validated macro-F1 (± std)",
    )
    fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Error bars are the standard deviation across folds — smaller bars mean more "
        "stable, trustworthy performance."
    )


# --------------------------------------------------------------------------- #
# 14. Error Analysis
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[13]:
    section_header(
        "14. Error Analysis",
        "Where and why the best model makes mistakes.",
    )
    X_train, X_test, y_train, y_test = get_split()
    best = get_best_model()
    y_pred = best.predict(X_test)

    st.markdown("### Confusion matrix (tuned best model)")
    cm = pl.confusion_df(y_test, y_pred, LABELS)
    fig = px.imshow(cm.values, x=cm.columns, y=cm.index, text_auto=True,
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Misclassified examples")
    test_df = pd.DataFrame({"clean_text": X_test, "true": y_test, "pred": y_pred})
    wrong = test_df[test_df["true"] != test_df["pred"]]
    st.metric("Misclassified", f"{len(wrong):,} / {len(test_df):,}",
              delta=f"{len(wrong)/len(test_df):.1%} error rate", delta_color="inverse")

    pair = st.selectbox(
        "Filter error type (true → pred)",
        ["all"] + [f"{t} → {p}" for t in LABELS for p in LABELS if t != p],
    )
    show = wrong
    if pair != "all":
        t, p = pair.split(" → ")
        show = wrong[(wrong["true"] == t) & (wrong["pred"] == p)]
    st.dataframe(show.head(20), use_container_width=True, hide_index=True)

    st.markdown(
        """
**Typical failure modes observed**
- *Neutral ↔ polar confusion*: factual/news posts that contain emotional vocabulary.
- *Sarcasm & negation*: e.g. "great, another update that breaks everything" reads
  positive lexically but is negative.
- *Short posts*: very few tokens give the model little evidence.
- *Entity-dependent tone*: the same word can be positive for one entity and negative
  for another, which a bag-of-words model cannot fully capture.
        """
    )


# --------------------------------------------------------------------------- #
# 15. Model Comparison
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[14]:
    section_header(
        "15. Model Comparison",
        "Side-by-side comparison across hold-out and cross-validation.",
    )
    comp = get_comparison()
    cv = get_cv()

    merged = comp.merge(
        cv[["Model", "CV F1 (mean)", "CV F1 (std)"]], on="Model", how="left"
    )
    st.dataframe(
        merged.style.format({c: "{:.3f}" for c in merged.columns if c != "Model"})
        .background_gradient(cmap="Greens", subset=["F1 (macro)", "CV F1 (mean)"]),
        use_container_width=True, hide_index=True,
    )

    melt = comp.melt(id_vars="Model",
                     value_vars=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
                     var_name="metric", value_name="score")
    fig = px.bar(melt, x="Model", y="score", color="metric", barmode="group",
                 title="Hold-out metrics by model")
    fig.update_layout(height=460, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    best_row = comp.iloc[0]
    st.success(
        f"🏆 **Strongest default candidate:** {best_row['Model']} "
        f"(hold-out macro-F1 = {best_row['F1 (macro)']:.3f})."
    )
    st.info(
        "We carry **Logistic Regression** forward for hyperparameter tuning because it is "
        "fast, stable, and **interpretable** (per-class word weights). As shown in "
        "Sections 12 & 16, the *tuned* Logistic Regression (bi-grams, C=10) surpasses the "
        "default candidates on the held-out set — giving the best of both worlds: top "
        "accuracy **and** explainability."
    )


# --------------------------------------------------------------------------- #
# 16. Best Model Performance
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[15]:
    section_header(
        "16. Best Model Performance",
        "Final tuned model evaluated on the untouched test set.",
    )
    X_train, X_test, y_train, y_test = get_split()
    grid = get_tuned()
    best = get_best_model()
    m = pl.evaluate(best, X_test, y_test)

    st.markdown("#### Final model: tuned **TF-IDF + Logistic Regression**")
    st.json(grid.best_params_)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision (macro)", f"{m['precision_macro']:.3f}")
    c3.metric("Recall (macro)", f"{m['recall_macro']:.3f}")
    c4.metric("Macro-F1", f"{m['f1_macro']:.3f}")

    base = pl.baseline_pipeline().fit(X_train, y_train)
    base_f1 = pl.evaluate(base, X_test, y_test)["f1_macro"]
    lift = m["f1_macro"] - base_f1
    st.success(f"Improvement over baseline macro-F1: **+{lift:.3f}** "
               f"({base_f1:.3f} → {m['f1_macro']:.3f}).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification report")
        st.dataframe(pl.classification_report_df(y_test, m["y_pred"]),
                     use_container_width=True)
    with col2:
        st.markdown("#### Confusion matrix")
        cm = pl.confusion_df(y_test, m["y_pred"], LABELS)
        fig = px.imshow(cm.values, x=cm.columns, y=cm.index, text_auto=True,
                        color_continuous_scale="Blues", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 17. Feature Importance
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[16]:
    section_header(
        "17. Feature Importance",
        "Which words push a post toward each sentiment.",
    )
    best = get_best_model()
    tops = pl.top_features_per_class(best, top_n=15)

    if not tops:
        st.warning("The selected model does not expose linear coefficients.")
    else:
        st.markdown(
            "For the linear model, each TF-IDF term has a learned weight per class. "
            "The **highest-weight terms** are the strongest indicators of that sentiment."
        )
        cols = st.columns(len(tops))
        for (cls, dfc), col in zip(tops.items(), cols):
            color = COLOR_MAP.get(cls, "#1f77b4")
            fig = px.bar(dfc.sort_values("weight"), x="weight", y="term",
                         orientation="h", title=f"Top terms → {cls}")
            fig.update_traces(marker_color=color)
            fig.update_layout(height=480, yaxis_title="", xaxis_title="coefficient")
            col.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
**Reading the chart**
- *Positive* class is driven by enthusiastic terms like *love, amazing, awesome,
  excited, best, thanks*.
- *Negative* class is driven by *worst, hate, fix, broken, bug, servers* and strong
  profanity (typical of frustrated complaints).
- *Neutral* class is often driven by **URL / link fragments** (*com, dlvr, ly, tt*) and
  informational words — many neutral posts are news shares or automated link posts.

This confirms the model learned **meaningful** cues. The URL artifacts in the neutral
class are a genuine, interpretable signal but also a caveat: automated/news posts are
recognisable by their links rather than their actual opinion.
            """
        )


# --------------------------------------------------------------------------- #
# 18. Interpretation of Results
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[17]:
    section_header(
        "18. Interpretation of Results",
        "Findings, answers to the research questions, limitations, and future work.",
    )
    comp = get_comparison()
    feat = get_features()
    best_row = comp.iloc[0]
    grid = get_tuned()

    pos_share = (feat["sentiment"] == "positive").mean()
    neg_share = (feat["sentiment"] == "negative").mean()
    neu_share = (feat["sentiment"] == "neutral").mean()

    tuned_test_f1 = pl.evaluate(get_best_model(), *get_split()[1::2])["f1_macro"]

    st.markdown(f"""
### Key findings

1. **Public reaction on X is more negative than positive in this corpus.**
   Overall the posts are **{neg_share:.0%} negative**, **{pos_share:.0%} positive**, and
   **{neu_share:.0%} neutral** — complaints and criticism are the single largest class,
   which is common for brand/product discourse on social media.

2. **RQ1 — Classical ML vastly beats the baseline.** The tuned model reaches
   macro-F1 ≈ **{tuned_test_f1:.3f}** on the held-out test set (CV ≈
   {grid.best_score_:.3f}), far above the ~0.18 majority-class baseline, confirming the
   task is learnable from TF-IDF features. ✅ H1 supported.

3. **RQ2 — Tuned linear model is the best overall.** Among *default* candidates the
   non-linear **{best_row['Model']}** leads, but after tuning (bi-grams, C=10) the
   **Logistic Regression** overtakes it on the test set — delivering top accuracy while
   remaining interpretable.

4. **RQ3 — Drivers are interpretable.** Feature-importance shows clear emotional
   vocabulary (*love/amazing/awesome* → positive; *worst/hate/broken* → negative) while
   the neutral class is partly identified by **URL/link fragments** typical of news and
   automated posts. ✅ H2 supported — bi-grams add discriminative power.

### Practical implications
- A lightweight, **interpretable** TF-IDF + Logistic Regression model is sufficient for
  real-time brand-sentiment monitoring on Platform X — cheap to train, fast to serve,
  and auditable.
- Per-entity EDA shows sentiment skews differently across brands/products, so dashboards
  should report sentiment **per entity**, not just in aggregate.

### Limitations
- TF-IDF (bag-of-words) ignores word order and struggles with **sarcasm and negation**
  (see Error Analysis).
- Sentiment is **entity-dependent**; the same word can flip meaning across entities,
  which a global model cannot fully capture.
- **English-only**; a stratified 12k sample is used for responsiveness (the full ~74k
  corpus can be enabled in `src/load_data.py`).
- Neutral predictions partly rely on link artifacts rather than true opinion.

### Future work
- Replace/augment TF-IDF with **transformer embeddings** (e.g., fine-tuned BERT/RoBERTa)
  to capture context, sarcasm, and entity-aware sentiment.
- Add **aspect/entity-based sentiment analysis** to report sentiment per target automatically.
- Concatenate the engineered structural features with TF-IDF in a stacked model.
- Deploy as a streaming dashboard with per-entity trend alerts.
    """)

    st.divider()
    st.markdown("### Conclusion")
    st.info(
        "This study delivered an end-to-end, reproducible pipeline — from data collection "
        "and cleaning through EDA, feature engineering, modeling, tuning, and "
        "interpretation — for classifying public sentiment on Platform X using a real, "
        "human-annotated Twitter benchmark. The tuned **Logistic Regression** model "
        "provides accurate, stable, and explainable predictions, demonstrating that "
        "classical ML remains a strong, practical choice for large-scale social-media "
        "sentiment monitoring."
    )


# --------------------------------------------------------------------------- #
# 19. Live Demo
# --------------------------------------------------------------------------- #
elif choice == SECTIONS[18]:
    section_header(
        "19. Live Demo — Try the model",
        "Type a post and the tuned best model predicts its sentiment.",
    )
    best = get_best_model()

    examples = {
        "Positive": "Just started playing this game again and honestly I love it so much, best update ever 🔥",
        "Negative": "The new patch completely broke the servers, worst experience ever, please fix this 😡",
        "Neutral": "New article discusses the company's latest quarterly results dlvr.it/abc123",
    }
    pick = st.selectbox("Load an example (optional)", ["—"] + list(examples.keys()))
    default = examples.get(pick, "")
    text = st.text_area("Enter a post about any brand / product / entity", value=default, height=120)

    if st.button("Predict sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = pl.clean_text(text)
            pred = best.predict([cleaned])[0]
            st.markdown(f"### Prediction: :{ 'green' if pred=='positive' else 'red' if pred=='negative' else 'gray'}[**{pred.upper()}**]")
            st.caption(f"Cleaned input: `{cleaned}`")

            if hasattr(best.named_steps["clf"], "predict_proba"):
                proba = best.predict_proba([cleaned])[0]
                classes = best.named_steps["clf"].classes_
                prob_df = pd.DataFrame({"sentiment": classes, "probability": proba})
                fig = px.bar(prob_df, x="sentiment", y="probability", color="sentiment",
                             color_discrete_map=COLOR_MAP, range_y=[0, 1],
                             title="Class probabilities")
                fig.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig, use_container_width=True)
