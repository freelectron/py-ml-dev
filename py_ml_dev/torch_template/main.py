import logging
import subprocess

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchinfo
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


class SomeDataset(Dataset):
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


class SomeModel(nn.Module):
    def __init__(
            self, dim_in, dim_out,
    ):
        super().__init__()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.some_layer = nn.Linear(dim_in, dim_out)
        self.activation = nn.ReLU()
        for l in [
            self.some_layer,
        ]:
            nn.init.normal_(l.weight, 0, 0.01)
        self.to(self.device)

    def forward(self, x):
        return  self.activation(self.some_layer(x))

def plot_loss(losses: list, stage_labels: list):
    fig, ax = plt.subplots(1, 1)
    for loss_list, label in zip(losses, stage_labels):
        ax.plot(list(range(len(loss_list))), loss_list, label=label)
        ax.legend()

    plt.show()

def save_and_log_model_summary(model, model_sum_file):
    with open(model_sum_file, "w") as f:
        f.write(str(torchinfo.summary(model, verbose=0)))
    mlflow.log_artifact(model_sum_file)


if __name__ == "__main__":
    git_commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )

    with mlflow.start_run(run_name=git_commit_hash[0:7]+"-template"):
        dim_in = 2
        X_train = np.random.rand(100,dim_in)
        y_train = np.random.rand(100,1)
        X_test = np.random.rand(100,dim_in)
        y_test = np.random.rand(100,1)
        train_dataloader = DataLoader(SomeDataset(X_train, y_train), batch_size=64)
        test_dataloader = DataLoader(SomeDataset(X_test, y_test), batch_size=64)

        # mlflow.set_tracking_uri("file:/"+os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow.pytorch.autolog()
        mlflow.set_experiment("template_experiment")

        n_epochs = 2
        n_factors = 5
        lr = 0.05
        mse = torch.nn.MSELoss()

        model = SomeModel(
            2, n_factors
        )
        model_sum_file = "../../../static/data/model_summary.txt"
        save_and_log_model_summary(model, model_sum_file)

        learning_params = dict(
            loss_func = mse,
            opt_func = SGD,
            lr = lr,
            metric = mse
        ) ; mlflow.log_params(learning_params)

        learn = BasicLearner(
            train_dataloader,
            test_dataloader,
            model=model,
            **learning_params
        )

        train_losses, test_losses, eval_metrics = learn.fit(n_epochs)
        plot_loss([train_losses, test_losses], ["train", "test"])

        """
        archive and deploy with torchserve 
        """