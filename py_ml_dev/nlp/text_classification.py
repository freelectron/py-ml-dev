import os
from datetime import datetime
from typing import List, Any, Tuple, Dict

import spacy
import ray
from py_ml_dev.utils.utils import load_pickle, save_pickle
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA, SparsePCA
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> List[str]:
    """
    Tokenize and clean a text string using spaCy.
    Removes stopwords, punctuation, and numbers. Lemmatizes tokens.
    """
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num
    ]


def clean_corpus(corpus: List[str], use_ray: bool = False) -> List[List[str]]:
    """
    Clean a list of text documents. Optionally parallelize with Ray.
    # """
    if use_ray:
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count() - 1)
        @ray.remote
        def remote_clean(text: str) -> List[str]:
            return clean_text(text)
        cleaned = ray.get([remote_clean.remote(text) for text in corpus])
        ray.shutdown()
        return cleaned
    else:
        return [clean_text(text) for text in corpus]


def vectorize_corpus(
        cleaned_corpus: List[List[str]], vocabulary: set = None
) -> Tuple[CountVectorizer, Any]:
    """
    Fit a CountVectorizer on the cleaned corpus and return the vectorizer and counts.
    """
    if vocabulary is None:
        vocabulary = set(token for doc in cleaned_corpus for token in doc)
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    vectorizer.fit([" ".join(tokens) for tokens in cleaned_corpus])
    counts = vectorizer.transform([" ".join(tokens) for tokens in cleaned_corpus])
    return vectorizer, counts


def tfidf_transform_(counts: Any) -> Any:
    """Transform count vectors to TF-IDF features."""
    return TfidfTransformer().fit_transform(counts)


def logistic_regression_cv() -> LogisticRegressionCV:
    """Train LogisticRegressionCV on the data."""
    return LogisticRegressionCV()


def xgboost_random_search(
        param_dist: Dict[str, List[Any]], n_iter: int = 10
) -> RandomizedSearchCV:
    """
    Train XGBoost with RandomizedSearchCV.
    """
    xgb = XGBClassifier()
    return RandomizedSearchCV(xgb, param_dist, n_iter=n_iter, n_jobs=-1)


def evaluate_model(
        model, X, y, label: str = ""
) -> float:
    """Evaluate a model and print accuracy."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1_s = f1_score(y, preds, average = "weighted")

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    displ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(1,1, figsize=(12, 12))
    ax.set_title(f"Accuracy for {label or model.__class__.__name__}: {acc:.4f}")
    ax.text(0.5, 1.02, f"The weighted f1 score is {f1_s}", transform=ax.transAxes, ha="center", fontsize=12)
    displ.plot(ax=ax)
    plt.show()

    return acc

def train_cv(estimator: BaseEstimator, X: Any, y: Any) -> BaseEstimator:
    start_time = datetime.now()
    trained_estimator = estimator.fit(X, y)
    print(f"Training {estimator.__class__} took", (datetime.now() - start_time).seconds, "seconds")
    evaluate_model(trained_estimator, X, y, "LogisticRegressionCV (train)")

    return trained_estimator

def main() -> None:
    # Load dataset
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    X_train, X_test, y_train, y_test = train_test_split(
        newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
    )
    # Clean training data (with caching)
    path_X_train_cleaned = "../../static/data/X_train_cleaned.npy"
    if os.path.exists(path_X_train_cleaned):
        X_train_cleaned = load_pickle(path_X_train_cleaned)
        print(f"Loaded cleaned training data from {path_X_train_cleaned}")
    else:
        start_time = datetime.now()
        X_train_cleaned = clean_corpus(X_train, use_ray=True)
        print(f"Cleaning training data took {(datetime.now() - start_time).seconds} seconds.")
        save_pickle(X_train_cleaned, path_X_train_cleaned)
        print(f"Saved cleaned training data to {path_X_train_cleaned}")

    # Vectorization
    vocabulary = set(token for doc in X_train_cleaned for token in doc)
    vectorizer, X_train_counts = vectorize_corpus(X_train_cleaned, vocabulary)

    # Preprocessing & model training
    n_components_pca = 10000
    pipeline_preprocess = make_pipeline(
        TfidfTransformer(),
        PCA(n_components=n_components_pca),
    )
    X_train_preprocessed = pipeline_preprocess.fit_transform(X_train_counts)

    xgb_param_dict = {
        "max_depth": [3],
        "min_child_weight": [4],
        "n_estimators": [50, 100],
    }
    log_reg = logistic_regression_cv()
    log_reg_best = train_cv(log_reg, X_train_preprocessed.toarray(), y_train)
    xg_boost_rand_search = xgboost_random_search(xgb_param_dict)
    xg_boost_cv_result = train_cv(xg_boost_rand_search, X_train_preprocessed.toarray(), y_train)
    # FixMe: interfaces do not match
    xg_boost_best = xg_boost_cv_result.best_estimator_

    # Preprocess test data
    X_test_cleaned = clean_corpus(X_test, use_ray=True)
    X_test_counts = vectorizer.transform([" ".join(tokens) for tokens in X_test_cleaned])
    X_test_preprocessed = pipeline_preprocess.fit_transform(X_test_counts)

    # Evaluate on test set
    evaluate_model(log_reg_best, X_test_preprocessed.toarray(), y_test, "LogisticRegressionCV (test)")
    evaluate_model(xg_boost_best, X_test_preprocessed.toarray(), y_test, "XGBClassifier (test)")


if __name__ == "__main__":
    main()