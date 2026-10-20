import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(y_true, y_pred, labels) -> dict:
    """Accuracy, per-class and macro F1 and the confusion matrix (rows true, columns predicted, in `labels` order)."""
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": dict(zip(labels, (float(x) for x in per_class))),
        "macro_f1": float(np.mean(per_class)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }
