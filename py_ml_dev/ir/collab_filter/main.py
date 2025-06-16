import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - %(levelname)s | %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("ir.collab_filtering")

class MovieLens100K(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.X = torch.Tensor(X).to(self.device)
        self.y = torch.Tensor(y).to(self.device)

    def __len__(self):
        assert self.X.size(0) == self.y.size(0)
        return self.y.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class BasicLearner:
    def __init__(
        self, dataloader_train, data_loader_test, model, loss_func, opt_func, lr, metric
    ):
        self.model = model
        self.dl_train = dataloader_train
        self.dl_test = data_loader_test
        self.loss_func = loss_func
        self.optim = opt_func(model.parameters(), lr)
        self.lr = lr
        self.metric = metric

    def fit(self, epochs):
        train_losses = list()
        test_losses = list()
        eval_metrics = list()
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            test_loss, metric = self.test_epoch()
            test_losses.append(test_loss)
            eval_metrics.append(metric)

            logger.info(
                f"Epoch {epoch} train loss = {train_loss} | test loss = {test_loss} ; metric = {metric} "
            )

        return train_losses, test_losses, eval_metrics

    def train_epoch(self):
        losses = list()
        self.model.train()
        for x, y in self.dl_train:
            self.optim.zero_grad()
            loss = self.calculate_grads(x, y)
            losses.append(loss.detach().cpu())
            self.optim.step()

        return torch.stack(losses).mean()

    def test_epoch(self):
        losses = list()
        val_metrics = list()
        self.model.eval()
        with torch.no_grad():
            for x, y in self.dl_test:
                preds = self.model(x)
                loss = self.loss_func(preds, y)
                losses.append(loss.detach().cpu())
                metric = self.calculate_eval_metric(preds, y)
                val_metrics.append(metric.detach().cpu())

        return torch.stack(losses).mean(), torch.stack(val_metrics).mean()

    def calculate_grads(self, x, y):
        preds = self.model(x)
        loss = self.loss_func(preds, y)
        loss.backward()

        return loss

    def calculate_eval_metric(self, preds, y):
        metric = self.metric(preds, y) if self.metric is not None else None

        return metric


class Perceptron(nn.Module):
    def __init__(
        self, num_users, dim_user_factors, num_movies, dim_movie_factors, y_range
    ):
        super().__init__()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.user_factors = nn.Embedding(num_users, dim_user_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_factors = nn.Embedding(num_movies, dim_movie_factors)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.y_range = y_range
        self.activation = nn.Sigmoid()
        for emb in [
            self.user_factors,
            self.user_bias,
            self.movie_bias,
            self.movie_factors,
        ]:
            nn.init.normal_(emb.weight, 0, 0.01)
        self.to(self.device)

    def forward(self, x):
        user_embeddings = self.user_factors(
            x[:, 0].to(torch.int64)
        )  # batch_size x n_user_factors
        movie_embeddings = self.movie_factors(
            x[:, 1].to(torch.int64)
        )  # batch_size x n_movie_factors
        user_bias_embedding = self.user_bias(x[:, 0].to(torch.int64))
        movie_bias_embedding = self.movie_bias(x[:, 1].to(torch.int64))
        res = (
            (user_embeddings * movie_embeddings).sum(dim=1)
            + user_bias_embedding.reshape(-1)
            + movie_bias_embedding.reshape(-1)
        )

        return (
            self.activation(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        )


def plot_loss(losses: list, stage_labels: list):
    fig, ax = plt.subplots(1, 1)
    for loss_list, label in zip(losses, stage_labels):
        ax.plot(list(range(len(loss_list))), loss_list, label=label)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    ratings = pd.read_csv(
        "../../../static/data/movielens/ml-100k/u.data",
        delimiter="\t",
        header=None,
        names=["user", "movie", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        "../../../static/data/movielens/ml-100k/u.item",
        delimiter="|",
        encoding="latin-1",
        usecols=(0, 1),
        names=("movie", "title"),
        header=None,
    )
    users = pd.read_csv(
        "../../../static/data/movielens/ml-100k/u.user",
        delimiter="|",
        header=None,
        usecols=(0, 1, 2),
        names=("user", "age", "gender"),
    )
    ratings_original = ratings.copy(deep=True)
    ratings = ratings.merge(movies, on="movie").merge(users, on="user")

    le_user = LabelEncoder()
    le_movie = LabelEncoder()

    ratings["userId"] = le_user.fit_transform(ratings["user"].values)
    ratings["titleId"] = le_movie.fit_transform(ratings["title"].values)

    df_train, df_test = train_test_split(
        ratings[["userId", "titleId", "rating"]], test_size=0.2, random_state=3
    )

    X_train = df_train[["userId", "titleId"]].values
    y_train = df_train[["rating"]].values
    X_test = df_test[["userId", "titleId"]].values
    y_test = df_test[["rating"]].values
    train_dataloader = DataLoader(MovieLens100K(X_train, y_train), batch_size=64)
    test_dataloader = DataLoader(MovieLens100K(X_test, y_test), batch_size=64)

    with mlflow.start_run(name=):
        n_factors = 5
        mse = torch.nn.MSELoss()
        model = Perceptron(
            len(le_user.classes_), n_factors, len(le_movie.classes_), n_factors, (0, 5.5)
        )
        learn = BasicLearner(
            train_dataloader,
            test_dataloader,
            model=model,
            loss_func=mse,
            opt_func=SGD,
            lr=0.05,
            metric=mse,
        )
        train_losses, test_losses, eval_metrics = learn.fit(2)
        plot_loss([train_losses, test_losses], ["train", "test"])
