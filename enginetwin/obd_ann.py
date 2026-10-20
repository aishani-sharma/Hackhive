"""A small neural network on the OBD readings alone: shows what the car's own sensors can and cannot detect."""
import numpy as np
import torch
from torch import nn

from enginetwin.fault_features import FEATURE_NAMES, OBD_KEYS

OBD_FEATURE_NAMES = [n for n in FEATURE_NAMES if n.startswith("obd_")]
EPOCHS = 300
HIDDEN = (32, 16)


def obd_matrix(labelled_set) -> np.ndarray:
    """The eight OBD columns (seven readings and the total fuel trim), in the order of the full feature vector."""
    columns = [labelled_set.obd[k] for k in OBD_KEYS]
    columns.append(labelled_set.obd["short_term_fuel_trim_bank1"] + labelled_set.obd["long_term_fuel_trim_bank1"])
    return np.column_stack(columns).astype(float)


class ObdAnn:
    """A fitted network with `classes_`, `predict`, `predict_proba` and `n_parameters`."""

    def __init__(self, network, mean, std, classes):
        self.network, self.mean, self.std, self.classes_ = network, mean, std, classes

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def predict_proba(self, X) -> np.ndarray:
        x = torch.tensor((np.asarray(X, dtype=float) - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self.network(x), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_obd_ann(X, y, seed=0) -> ObdAnn:
    """Two small hidden layers (about 900 weights) trained full-batch on standardised readings."""
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=float)
    classes, target = np.unique(np.asarray(y), return_inverse=True)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    x = torch.tensor((X - mean) / std, dtype=torch.float32)
    t = torch.tensor(target, dtype=torch.long)
    sizes = (X.shape[1],) + HIDDEN
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    network = nn.Sequential(*layers, nn.Linear(sizes[-1], len(classes)))
    optimiser = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(EPOCHS):
        optimiser.zero_grad()
        nn.functional.cross_entropy(network(x), t).backward()
        optimiser.step()
    return ObdAnn(network, mean, std, classes)
