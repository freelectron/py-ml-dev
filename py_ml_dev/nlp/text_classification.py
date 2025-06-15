import os
import pickle
import subprocess
from datetime import datetime
from typing import List, Any, Tuple, Dict

import pandas as pd
import spacy
import ray
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


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


def save_pickle(obj: Any, path: str) -> None:
    """Save an object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


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


def tfidf_transform(counts: Any) -> Any:
    """Transform count vectors to TF-IDF features."""
    return TfidfTransformer().fit_transform(counts)


def train_logistic_regression(X, y) -> LogisticRegressionCV:
    """Train LogisticRegressionCV on the data."""
    clf = LogisticRegressionCV()
    clf.fit(X, y)
    return clf


def train_xgboost_with_search(
        X, y, param_dist: Dict[str, List[Any]], n_iter: int = 10
) -> RandomizedSearchCV:
    """
    Train XGBoost with RandomizedSearchCV.
    """
    xgb = XGBClassifier()
    search = RandomizedSearchCV(xgb, param_dist, n_iter=n_iter, n_jobs=-1)
    search.fit(X, y)
    return search


def evaluate_model(
        model, X, y, label: str = ""
) -> float:
    """Evaluate a model and print accuracy."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    # see f1 score too
    print(f"Accuracy for {label or model.__class__.__name__}: {acc:.4f}")
    cm = confusion_matrix(y, preds)
    displ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(1,1, figsize=(12, 12))
    ax.set_title(f"Accuracy for {label or model.__class__.__name__}: {acc:.4f}")
    displ.plot(ax=ax)
    mlflow.log_figure(fig, f"confusion_matrix.png")
    plt.show()

    return acc

def plot_class_frequencies(y: List[int]) -> None:
    """Plot class frequencies in the dataset."""
    class_counts = pd.Series(y).value_counts()
    plt.bar(class_counts.index, class_counts.values)
    plt.xticks(class_counts.index, rotation=45)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Frequencies")
    plt.show()

def main(name="simple_text_classification") -> None:
    # Load dataset
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    X_train, X_test, y_train, y_test = train_test_split(
        newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
    )
    plot_class_frequencies(newsgroups.target)

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

    # Preprocessing train
    vocabulary = set(token for doc in X_train_cleaned for token in doc)
    vectorizer, X_train_counts = vectorize_corpus(X_train_cleaned, vocabulary)
    X_train_tfidf = tfidf_transform(X_train_counts)
    n_components = 7000
    pca_estimator = PCA(n_components=n_components).fit(X_train_tfidf.toarray())
    X_train_tfidf = pca_estimator.transform(X_train_tfidf.toarray())
    print(f"Reduced training data to {n_components} dimensions using PCA. "
          f"Explained variance ratio: {pca_estimator.explained_variance_ratio_.sum():.4f}")

    # Preprocess test data
    X_test_cleaned = clean_corpus(X_test, use_ray=True)
    X_test_counts = vectorizer.transform([" ".join(tokens) for tokens in X_test_cleaned])
    X_test_tfidf = tfidf_transform(X_test_counts)
    X_test_tfidf = pca_estimator.transform(X_test_tfidf.toarray())

    with mlflow.start_run(run_name=name+"-log-reg", log_system_metrics=True):
        # Train Logistic Regression
        mlflow.log_param("n_components", n_components)
        start_time = datetime.now()
        clf_logistic = train_logistic_regression(X_train_tfidf, y_train)
        print("Training Logistic Regression with CV took", (datetime.now() - start_time).seconds, "seconds")
        evaluate_model(clf_logistic, X_train_tfidf, y_train, "LogisticRegressionCV (train)")
        evaluate_model(clf_logistic, X_test_tfidf, y_test, "LogisticRegressionCV (test)")

    with mlflow.start_run(run_name=name+"-xgboost", log_system_metrics=True):
        mlflow.log_param("n_components", n_components)
        # Train XGBoost with RandomizedSearchCV
        start_time = datetime.now()
        param_dist = {
            "max_depth": [3],
            "min_child_weight": [5, 7],
            "n_estimators": [75],
            "gamma": [4, 6, 8],
        }
        xgb_search = train_xgboost_with_search(X_train_tfidf, y_train, param_dist)
        print("Training XGBoost with RandomizedSearchCV took", (datetime.now() - start_time).seconds, "seconds")
        evaluate_model(xgb_search.best_estimator_, X_train_tfidf, y_train, "XGBoostSearch (train)")
        evaluate_model(xgb_search.best_estimator_, X_test_tfidf, y_test, "XGBClassifier (test)")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    mlflow.autolog()
    mlflow.set_experiment("sklearn_simple_text_classification")
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    name=f"{git_commit_hash[:7]}"

    main(name=name)