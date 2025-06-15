import logging

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

ratings = pd.read_csv("../../../static/data/movielens/ml-100k/u.data", delimiter='\t', header=None, names=['user','movie','rating','timestamp'])
movies = pd.read_csv("../../../static/data/movielens/ml-100k/u.item",  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
users = pd.read_csv("../../../static/data/movielens/ml-100k/u.user", delimiter="|", header=None, usecols=(0,1,2), names=("user", "age", "gender"))
ratings_original = ratings.copy(deep=True)
ratings = ratings.merge(movies, on='movie').merge(users, on="user")

le_user = LabelEncoder()
le_movie = LabelEncoder()

ratings["userId"] = le_user.fit_transform(ratings["user"].values)
ratings["titleId"] = le_movie.fit_transform(ratings["title"].values)

df_train, df_test = train_test_split(ratings[["userId", "titleId", "rating"]], test_size=0.2, random_state=3)


class MovieLens100K(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

train_dataloader = DataLoader(MovieLens100K(torch.Tensor(df_train[["userId", "titleId"]].values),torch.Tensor(df_train[["rating"]].values)), batch_size=64)
test_dataloader = DataLoader(MovieLens100K(torch.Tensor(df_test[["userId", "titleId"]].values),torch.Tensor(df_test[["rating"]].values)), batch_size=64)

class BasicLearner:
    def __init__(self, dataloader_train, data_loader_test, model, loss_func, opt_func, lr, metric):
        self.model = model
        self.dl_train = dataloader_train
        self.dl_test = data_loader_test
        self.loss_func = loss_func
        self.optim = opt_func(model.parameters(), lr)
        self.lr = lr
        self.metric = metric

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            test_loss, metric = self.test_epoch()
            logger.info(f"Epoch {epoch} train loss = {train_loss} | test loss = {test_loss} ; metric = {metric} ")

    def train_epoch(self):
        losses = list()
        self.model.train()
        for x, y in self.dl_train:
            self.optim.zero_grad()
            loss = self.calculate_grads(x, y)
            losses.append(loss)
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
                losses.append(loss)
                metric = self.calculate_eval_metric(preds, y)
                val_metrics.append(metric)

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
    def __init__(self, num_users, dim_user_factors, num_movies, dim_movie_factors, y_range):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, dim_user_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_factors = nn.Embedding(num_movies, dim_movie_factors)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.y_range = y_range
        self.activation = nn.Sigmoid()
        # TODO: add initialisation from normal distribution for all embedding layers

    def forward(self, x):
        user_embeddings = self.user_factors(x[:,0].to(torch.int64))    # batch_size x n_user_factors
        movie_embeddings = self.movie_factors(x[:,1].to(torch.int64))  # batch_size x n_movie_factors
        user_bias_embedding = self.user_bias(x[:,0].to(torch.int64))
        movie_bias_embedding = self.movie_bias(x[:, 1].to(torch.int64))
        res = (user_embeddings * movie_embeddings).sum(dim=1) + user_bias_embedding.reshape(-1) + movie_bias_embedding.reshape(-1)

        return self.activation(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

def mse(preds, targets):
    return ((targets.flatten() - preds.flatten())**2).mean()

n_factors = 5

model = Perceptron(len(le_user.classes_), n_factors,  len(le_movie.classes_), n_factors, (0, 5.5))
learn = BasicLearner(train_dataloader, test_dataloader, model=model, loss_func=mse, opt_func=SGD, lr=0.05, metric=mse)
learn.fit(10)
