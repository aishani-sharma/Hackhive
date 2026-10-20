"""Multimodal CNN: spectrogram images of sound and vibration plus OBD values, one sigmoid per fault (so two at once)."""
import numpy as np
import torch
from torch import nn

FAULT_NAMES = ("misfire", "knock", "valve_wear")


def faults_to_multihot(labels) -> np.ndarray:
    """(N, 3) flags for misfire, knock and valve_wear; a combined label like "misfire+knock" sets two flags."""
    return np.array([[float(f in str(label).split("+")) for f in FAULT_NAMES] for label in labels])


def labels_from_proba(proba, threshold=0.5) -> np.ndarray:
    """The 7 fault labels from per-fault probabilities. At most two faults are kept (the two most probable)."""
    labels = []
    for row in np.asarray(proba):
        on = [k for k in range(3) if row[k] > threshold]
        if len(on) == 3:
            on.remove(int(np.argmin(row)))
        labels.append("+".join(FAULT_NAMES[k] for k in on) or "healthy")
    return np.array(labels)


class _Net(nn.Module):
    def __init__(self, n_obd):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.obd = nn.Sequential(nn.Linear(n_obd, 16), nn.ReLU())
        self.head = nn.Linear(16 + 16 + 16, 3)

    def forward(self, spec, obd):
        z = self.conv(spec)
        pooled = torch.cat([z.amax(dim=(2, 3)), z.mean(dim=(2, 3))], dim=1)  # a short event anywhere, and the overall level
        return self.head(torch.cat([pooled, self.obd(obd)], dim=1))


class CnnModel:
    def __init__(self, net, spec_mean, spec_std, obd_mean, obd_std):
        self.net, self.spec_mean, self.spec_std, self.obd_mean, self.obd_std = net, spec_mean, spec_std, obd_mean, obd_std

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _inputs(self, spec, obd):
        spec = torch.tensor((np.asarray(spec, dtype=np.float32) - self.spec_mean) / self.spec_std)
        obd = torch.tensor(((np.asarray(obd, dtype=float) - self.obd_mean) / self.obd_std).astype(np.float32))
        return spec, obd

    def predict_proba(self, spec, obd, batch=256) -> np.ndarray:
        spec, obd = self._inputs(spec, obd)
        out = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(spec), batch):
                out.append(torch.sigmoid(self.net(spec[i : i + batch], obd[i : i + batch])).numpy())
        return np.concatenate(out)


def fit_cnn(spec, obd, multihot, seed=0, epochs=12, batch=64) -> CnnModel:
    torch.manual_seed(seed)
    spec, obd = np.asarray(spec, dtype=np.float32), np.asarray(obd, dtype=float)
    model = CnnModel(_Net(obd.shape[1]), float(spec.mean()), float(spec.std()) + 1e-9, obd.mean(axis=0), obd.std(axis=0) + 1e-9)
    x_spec, x_obd = model._inputs(spec, obd)
    target = torch.tensor(np.asarray(multihot), dtype=torch.float32)
    optimiser = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.net.train()
        order = torch.randperm(len(x_spec), generator=generator)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            optimiser.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model.net(x_spec[idx], x_obd[idx]), target[idx]).backward()
            optimiser.step()
    return model


def occlusion_map(model, spec, obd, fault, patch=8, stride=4) -> np.ndarray:
    """(64, 64) heat map: how much the fault's probability drops when a patch x patch square (both channels) is replaced
    by the spectrogram's average level. Each pixel holds the mean drop over the squares covering it; negative drops are 0."""
    spec = np.asarray(spec, dtype=np.float32)
    k = FAULT_NAMES.index(fault)
    base = model.predict_proba(spec[None], np.asarray(obd)[None])[0, k]
    h, w = spec.shape[1:]
    starts = [(r, c) for r in range(0, h - patch + 1, stride) for c in range(0, w - patch + 1, stride)]
    masked = np.repeat(spec[None], len(starts), axis=0)
    for n, (r, c) in enumerate(starts):
        masked[n, :, r : r + patch, c : c + patch] = spec.mean()
    drop = base - model.predict_proba(masked, np.repeat(np.asarray(obd)[None], len(starts), axis=0))[:, k]
    total, count = np.zeros((h, w)), np.zeros((h, w))
    for (r, c), d in zip(starts, drop):
        total[r : r + patch, c : c + patch] += d
        count[r : r + patch, c : c + patch] += 1
    return np.clip(total / np.maximum(count, 1), 0, None)
