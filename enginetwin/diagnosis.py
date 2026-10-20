"""One call from a window to a diagnosis: label, confidence, severity, reasons, unfamiliarity and a heat map."""
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from enginetwin.cnn import FAULT_NAMES, labels_from_proba, occlusion_map
from enginetwin.explain import reasons
from enginetwin.fault_features import extract_features
from enginetwin.noise import estimate_noise_db
from enginetwin.recording import has_obd
from enginetwin.schema import Window
from enginetwin.severity import estimate_severity
from enginetwin.spectrograms import standardise_spectrogram, window_spectrogram

FILES = ("model_cnn.pkl", "model_softmax.pkl", "model_autoencoder.pkl", "model_severity.pkl", "baseline.pkl")


@dataclass
class DiagnosisResult:
    label: str  # one of the seven fault labels, from the CNN
    status: str  # "Healthy" or "Fault"
    confidence: float  # fault: the weakest detected fault's probability; healthy: 1 - the strongest fault probability
    probabilities: dict  # CNN probability for each fault type
    severities: dict  # estimated severity (0 to 1) of each detected fault
    reasons: list  # plain-language sentences
    softmax_label: str  # the explainable baseline's seven-label opinion
    softmax_probabilities: dict
    anomaly: bool  # does the window look unfamiliar to the healthy-baseline autoencoder at this noise level?
    anomaly_score: float
    anomaly_threshold: float
    noise_db: float  # noise-to-signal ratio estimated from the recording (higher is noisier)
    obd_available: bool
    features: np.ndarray = field(repr=False, default=None)


class Diagnoser:
    def __init__(self, cnn, softmax, autoencoder, thresholds, severity_models, baseline):
        self.cnn, self.softmax, self.autoencoder = cnn, softmax, autoencoder
        self.thresholds, self.severity_models, self.baseline = thresholds, severity_models, baseline

    def _inputs(self, window: Window):
        vector, _ = extract_features(window)
        return vector, vector[-8:][None]  # the last eight features are the OBD values

    def diagnose(self, window: Window) -> DiagnosisResult:
        vector, obd = self._inputs(window)
        spec = standardise_spectrogram(window_spectrogram(window))[None]
        proba = self.cnn.predict_proba(spec, obd)[0]
        label = labels_from_proba(proba[None])[0]
        detected = label.split("+") if label != "healthy" else []
        probabilities = dict(zip(FAULT_NAMES, (float(p) for p in proba)))
        confidence = min(probabilities[f] for f in detected) if detected else 1.0 - max(probabilities.values())
        soft = self.softmax.predict_proba(vector[None])[0]
        noise_db = estimate_noise_db(window)
        score = float(self.autoencoder.score(vector[None])[0])
        limit = float(self.thresholds.threshold_for(noise_db))
        return DiagnosisResult(
            label=label,
            status="Fault" if detected else "Healthy",
            confidence=float(confidence),
            probabilities=probabilities,
            severities={f: estimate_severity(self.severity_models, vector, f) for f in detected},
            reasons=reasons(vector, label, self.baseline),
            softmax_label=str(self.softmax.classes_[soft.argmax()]),
            softmax_probabilities=dict(zip((str(c) for c in self.softmax.classes_), (float(p) for p in soft))),
            anomaly=score > limit,
            anomaly_score=score,
            anomaly_threshold=limit,
            noise_db=float(noise_db),
            obd_available=has_obd(window),
            features=vector,
        )

    def heatmap(self, window: Window, fault: str) -> np.ndarray:
        """(64, 64) occlusion map: which part of the spectrogram the CNN relied on for `fault`."""
        _, obd = self._inputs(window)
        spec = standardise_spectrogram(window_spectrogram(window))
        return occlusion_map(self.cnn, spec, obd[0], fault)

    def save(self, cache_dir) -> None:
        cache = Path(cache_dir)
        parts = (
            self.cnn, self.softmax, {"model": self.autoencoder, "thresholds": self.thresholds}, self.severity_models, self.baseline,
        )
        for name, part in zip(FILES, parts):
            with open(cache / name, "wb") as f:
                pickle.dump(part, f)

    @classmethod
    def load(cls, cache_dir) -> "Diagnoser":
        cache = Path(cache_dir)
        cnn, softmax, ae, severity, baseline = (pickle.load(open(cache / name, "rb")) for name in FILES)
        return cls(cnn, softmax, ae["model"], ae["thresholds"], severity, baseline)
