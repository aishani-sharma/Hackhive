from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class _Model:
    """A fitted scaler + classifier with `classes_`, `predict` and `predict_proba`."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes_ = pipeline.classes_

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def fit_svm(X, y, seed=0) -> _Model:
    """RBF support vector machine on standardised features, with calibrated class probabilities."""
    return _Model(make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf", C=10.0, random_state=seed), ensemble=False)).fit(X, y))


def fit_softmax(X, y, seed=0) -> _Model:
    """Multinomial logistic (softmax) regression on standardised features: one linear weight per feature and class."""
    return _Model(make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000, random_state=seed)).fit(X, y))


def top_features(model, names, label, n=5) -> list:
    """The `n` features with the largest-magnitude softmax weight for `label`, as (name, weight) pairs. A positive
    weight means a higher value of that feature (in standard deviations) pushes towards `label`."""
    row = list(model.classes_).index(label)
    weights = model.pipeline[-1].coef_[row]
    order = sorted(range(len(names)), key=lambda i: -abs(weights[i]))[:n]
    return [(names[i], float(weights[i])) for i in order]
