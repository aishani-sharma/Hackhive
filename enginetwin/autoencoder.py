"""Healthy-baseline autoencoder: trained on healthy windows only, so anything it cannot rebuild is unfamiliar."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class Autoencoder:
    def __init__(self, network, mean, std):
        self.network, self.mean, self.std = network, mean, std

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def score(self, X) -> np.ndarray:
        """Mean squared reconstruction error per row, in units of the healthy data's own spread."""
        x = torch.tensor(((np.asarray(X, dtype=float) - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            return ((self.network(x) - x) ** 2).mean(dim=1).numpy().astype(float)


def fit_autoencoder(X_healthy, seed=0, epochs=300) -> Autoencoder:
    """A narrow bottleneck forces the network to learn how healthy readings move together; full-batch training."""
    torch.manual_seed(seed)
    X = np.asarray(X_healthy, dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    x = torch.tensor(((X - mean) / std).astype(np.float32))
    dims = X.shape[1]
    bottleneck = max(3, dims // 5)
    network = nn.Sequential(
        nn.Linear(dims, 16), nn.ReLU(), nn.Linear(16, bottleneck), nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, dims)
    )
    optimiser = torch.optim.Adam(network.parameters(), lr=0.01)
    for _ in range(epochs):
        optimiser.zero_grad()
        nn.functional.mse_loss(network(x), x).backward()
        optimiser.step()
    return Autoencoder(network, mean, std)


def anomaly_threshold(healthy_scores, q=0.99) -> float:
    """The score below which a share `q` of healthy windows fall; anything above is flagged as unfamiliar."""
    return float(np.quantile(healthy_scores, q))


def anomaly_auroc(healthy_scores, fault_scores) -> float:
    """Chance that a fault window scores higher than a healthy one (0.5 = no better than guessing)."""
    scores = np.concatenate([healthy_scores, fault_scores])
    truth = np.concatenate([np.zeros(len(healthy_scores)), np.ones(len(fault_scores))])
    return float(roc_auc_score(truth, scores))


def healthy_threshold(model, X, fault_labels, q=0.99) -> float:
    """Alarm level from the healthy rows of a validation set (so faults in it never raise the bar)."""
    return anomaly_threshold(model.score(np.asarray(X)[np.asarray(fault_labels) == "healthy"]), q)
